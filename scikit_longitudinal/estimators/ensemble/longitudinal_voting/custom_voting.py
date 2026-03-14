# pylint: disable=W0222
from enum import Enum
from typing import Any, Callable, List, Optional

import numpy as np
from overrides import override
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from scikit_longitudinal.templates import CustomClassifierMixinEstimator


class TieBreaker(Enum):
    """An enum class for tie-breaking mechanisms in case of a tie in voting.

    Attributes:
        LAST: The last wave's prediction is used.
        FIRST: The first wave's prediction is used.
        RANDOM: A random prediction is used.

    """

    LAST = "last"
    FIRST = "first"
    RANDOM = "random"


class LongitudinalCustomVoting(CustomClassifierMixinEstimator):
    """A custom voting classifier duplicate to the one in scikit-learn.

    The only difference is that this class allows for ensemble learning using a custom longitudinal voting strategy. It
    aggregates predictions from multiple classifiers and determines the final output based on the specified strategy.
    Another difference is that the tie-breaking mechanism favours the last wave's prediction by default.

    To see the strategy in action, check out the `LongitudinalVoting` class.

    The classifier supports binary and multiclass hard voting. `predict_proba` returns normalised vote shares across
    the fitted `classes_`, ensuring that `predict` always returns true class labels instead of raw argmax indices.

    Attributes:
        estimators (List[CustomClassifierMixinEstimator]):
            A list of classifiers for the ensemble.

        weights (Optional[List[float]]):
            Weights for each estimator for weighted voting (default is None).

        tie_breaker (TieBreaker):
            The tie-breaking mechanism to resolve ties in voting (default is TieBreaker.LAST).

        extract_wave (Callable):
            A function to extract specific wave data for training (default is None). When provided, estimator order is
            treated as wave order.

    Raises:
        ValueError: If no estimators are provided or if weights have incorrect length.
        NotFittedError: If attempting to predict before fitting the model.

    """

    def __init__(
        self,
        estimators: List[CustomClassifierMixinEstimator],
        weights: Optional[List[float]] = None,
        tie_breaker: TieBreaker = TieBreaker.LAST,
        extract_wave: Callable = None,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.tie_breaker = tie_breaker
        self.extract_wave = extract_wave
        self.classes_ = None
        self._fitted = False

    @override
    def _fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LongitudinalCustomVoting":
        """Fits the ensemble model.

        Args:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.

        Returns:
            LongitudinalCustomVoting:
                The fitted ensemble model.

        Raises:
            ValueError: If no estimators are provided or if weights have incorrect length.

        """
        _ = sample_weight

        if self.estimators:
            for _, estimator in self.estimators:
                check_is_fitted(
                    estimator, msg="Estimators must be fitted before using this method."
                )
        else:
            raise ValueError("No estimators were provided.")

        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if weights.ndim != 1 or len(weights) != len(self.estimators):
                raise ValueError(
                    f"The length of weights must match the number of estimators ({len(self.estimators)})."
                )
            if np.any(weights < 0):
                raise ValueError("Voting weights must be non-negative.")
            if np.isclose(weights.sum(), 0.0):
                raise ValueError("Voting weights must sum to a positive value.")
            self.weights = weights
        else:
            self.weights = np.ones(len(self.estimators), dtype=float)

        self.classes_ = unique_labels(y)
        self._fitted = True
        return self

    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts using the ensemble model.

        Args:
            X (np.ndarray):
                The test data.

        Returns:
            np.ndarray:
                The predicted values.

        Raises:
            NotFittedError: If attempting to predict before fitting the model.
            ValueError: If an estimator predicts labels that were not seen during fitting.

        """
        if not self._fitted:
            raise NotFittedError("Ensemble model is not fitted yet.")

        predictions = self._collect_predictions(X)
        vote_counts = self._compute_vote_counts(predictions)
        winning_indices = np.argmax(vote_counts, axis=1)
        winning_labels = self.classes_[winning_indices].astype(object)

        max_vote = np.max(vote_counts, axis=1)[:, None]
        ties = np.sum(np.isclose(vote_counts, max_vote), axis=1) > 1
        if np.any(ties):
            for sample_index in np.flatnonzero(ties):
                tied_class_indices = np.flatnonzero(
                    np.isclose(vote_counts[sample_index], max_vote[sample_index])
                )
                tied_labels = self.classes_[tied_class_indices]
                winning_labels[sample_index] = self._break_tie(
                    predictions[:, sample_index], tied_labels
                )

        return np.asarray(winning_labels, dtype=self.classes_.dtype)

    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probabilities using the ensemble model.

        The returned values are normalised vote shares derived from hard predictions, rather than averaged base
        estimator confidence scores.

        Args:
            X (np.ndarray):
                The test data.

        Returns:
            np.ndarray:
                The predicted probabilities.

        Raises:
            NotFittedError: If attempting to predict before fitting the model.

        """
        if not self._fitted:
            raise NotFittedError("Ensemble model is not fitted yet.")

        vote_counts = self._compute_vote_counts(self._collect_predictions(X))
        total_weight = float(np.sum(self.weights))
        if np.isclose(total_weight, 0.0):
            raise ValueError("Voting weights must sum to a positive value.")
        return vote_counts / total_weight

    def _extract_wave(self, X: np.ndarray, estimator_index: int) -> np.ndarray:
        """Reduce the input data to a specific wave if a wave extractor is provided.

        Args:
            X (np.ndarray):
                The input data.
            estimator_index (int):
                The index of the estimator whose wave should be extracted.

        Returns:
            np.ndarray: The extracted data.

        """
        if self.extract_wave:
            return X[:, self.extract_wave(estimator_index, True)[2]]
        return X

    def _collect_predictions(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                estimator.predict(self._extract_wave(X, estimator_index))
                for estimator_index, (_, estimator) in enumerate(self.estimators)
            ],
            dtype=object,
        )

    def _compute_vote_counts(self, predictions: np.ndarray) -> np.ndarray:
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        vote_counts = np.zeros((predictions.shape[1], len(self.classes_)), dtype=float)

        for estimator_index, estimator_predictions in enumerate(predictions):
            try:
                encoded_predictions = np.asarray(
                    [
                        class_to_index[prediction]
                        for prediction in estimator_predictions
                    ],
                    dtype=int,
                )
            except KeyError as error:
                raise ValueError(
                    f"Estimator predicted label {error.args[0]!r}, which was not observed during fitting."
                ) from error

            np.add.at(
                vote_counts,
                (np.arange(predictions.shape[1]), encoded_predictions),
                self.weights[estimator_index],
            )

        return vote_counts

    def _break_tie(self, votes: np.ndarray, tied_labels: np.ndarray) -> Any:
        """Breaks a tie using the specified tie-breaker.

        Args:
            votes (np.ndarray): The votes to break the tie for.
            tied_labels (np.ndarray): The labels tied for the maximum vote count.

        Returns:
            Any: The selected value to break the tie.

        Raises:
            ValueError: If an invalid tie breaker is specified.

        """
        tied_labels = np.asarray(tied_labels, dtype=object)
        tied_set = set(tied_labels.tolist())

        if self.tie_breaker == TieBreaker.LAST:
            for vote in votes[::-1]:
                if vote in tied_set:
                    return vote
            return tied_labels[-1]
        if self.tie_breaker == TieBreaker.FIRST:
            for vote in votes:
                if vote in tied_set:
                    return vote
            return tied_labels[0]
        if self.tie_breaker == TieBreaker.RANDOM:
            return np.random.choice(tied_labels)
        raise ValueError(f"Invalid tie breaker: {self.tie_breaker}")
