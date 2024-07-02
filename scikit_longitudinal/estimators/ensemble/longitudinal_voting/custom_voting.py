# pylint: disable=W0222
from enum import Enum
from typing import Any, Callable, List, Optional

import numpy as np
from overrides import override
from sklearn.exceptions import NotFittedError
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

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    The only difference is that this class allows for ensemble learning using a custom longitudinal voting strategy. It
    aggregates predictions from multiple classifiers and determines the final output based on the specified strategy.
    Another difference is that the tie-breaking mechanism is favoring the last wave's prediction by default.

    To see the strategy in action, check out the `LongitudinalVoting` class.

    Attributes:
        estimators (List[CustomClassifierMixinEstimator]):
            A list of classifiers for the ensemble.

        weights (Optional[List[float]]):
            Weights for each estimator for weighted voting (default is None).

        tie_breaker (TieBreaker):
            The tie-breaking mechanism to resolve ties in voting (default is TieBreaker.LAST).

        extract_wave (Callable):
            A function to extract specific wave data for training (default is None).

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
        self._fitted = False

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "LongitudinalCustomVoting":
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
        if self.estimators:
            for _, estimator in self.estimators:
                check_is_fitted(estimator, msg="Estimators must be fitted before using this method.")
        else:
            raise ValueError("No estimators were provided.")

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(f"The length of weights must match the number of estimators ({len(self.estimators)}).")

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
            ValueError: If predictions are not numerical or not binary.

        """
        if not self._fitted:
            raise NotFittedError("Ensemble model is not fitted yet.")

        predictions = np.array(
            [
                estimator.predict(self._extract_wave(X, int(wave_number.split("_")[-1])))
                for wave_number, estimator in self.estimators
            ]
        )

        if not np.issubdtype(predictions.dtype, np.number):
            raise ValueError("Predictions must be numerical. We do not support categorical predictions yet.")
        if predictions.max() > 1:
            raise ValueError("Predictions must be binary.")

        if self.weights is not None:  # Weighted voting
            weighted_votes = np.average(predictions, axis=0, weights=self.weights)
            vote_counts = np.array([weighted_votes, 1 - weighted_votes])
        else:  # Majority voting
            vote_counts = np.apply_along_axis(self.majority_voting, axis=0, arr=predictions)

        # Apply tie-breaking mechanism and return the final predictions
        return np.array(
            [
                self._break_tie(predictions[:, i]) if self._is_tie(vote_counts[:, i]) else np.argmax(vote_counts[:, i])
                for i in range(predictions.shape[1])
            ]
        )

    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probabilities using the ensemble model.

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

        # Collect probability predictions from each classifier
        prob_predictions = np.array(
            [
                estimator.predict_proba(self._extract_wave(X, int(wave_number.split("_")[-1])))
                for wave_number, estimator in self.estimators
            ]
        )

        if self.weights is not None:  # Weighted average of probabilities
            return np.average(prob_predictions, axis=0, weights=self.weights)
        return np.mean(prob_predictions, axis=0)

    def _extract_wave(self, X: np.ndarray, wave: int) -> np.ndarray:
        """Reduce the input data to a specific wave if a wave extractor is provided.

        Args:
            X (np.ndarray):
                The input data.
            wave (int):
                The wave number to extract.

        Returns:
            np.ndarray: The extracted data.

        """
        if self.extract_wave:
            return X[:, self.extract_wave(wave, True)[2]]
        return X

    def _break_tie(self, votes: np.ndarray) -> Any:
        """Breaks a tie using the specified tie-breaker.

        Args:
            votes (np.ndarray): The votes to break the tie for.

        Returns:
            Any: The selected value to break the tie.

        Raises:
            ValueError: If an invalid tie breaker is specified.

        """
        if self.tie_breaker == TieBreaker.LAST:
            return votes[-1]
        if self.tie_breaker == TieBreaker.FIRST:
            return votes[0]
        if self.tie_breaker == TieBreaker.RANDOM:
            return np.random.choice(votes)
        raise ValueError(f"Invalid tie breaker: {self.tie_breaker}")

    @staticmethod
    def majority_voting(column: np.ndarray) -> np.ndarray:
        """Applies majority voting to a column of predictions.

        Attributes:
            column (np.ndarray):
                A column of predictions.

        Returns:
            np.ndarray:
                The majority vote.

        """
        return np.bincount(column, minlength=2)

    @staticmethod
    def _is_tie(votes: np.ndarray) -> bool:
        """Determines if there is a tie in voting.

        Attributes:
            votes (np.ndarray):
                The votes to check for a tie.

        Returns:
            bool:
                True if there is a tie, False otherwise.

        """
        return len(set(votes)) == 1
