# pylint: disable=E1101,W0222,R0801
from enum import Enum, auto
from typing import Callable, List

import numpy as np
from overrides import override
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from scikit_longitudinal.estimators.ensemble.longitudinal_voting.custom_voting import LongitudinalCustomVoting
from scikit_longitudinal.templates import CustomClassifierMixinEstimator


class LongitudinalEnsemblingStrategy(Enum):
    """
    An enum for the different longitudinal voting strategies.

    !!! note "Math Plugin Seems Capricious"
        We will sometime not "interpret" the math on purpose to avoid yielding `math error plugin`.

    Attributes:
        MAJORITY_VOTING (int):
            Simple consensus voting where the most frequent prediction is selected.
        DECAY_LINEAR_VOTING (int):
            Weights each classifier's vote based on the recency of its wave using a linear decay.
            Weight formula:

            ```math
            ( w_i = \\frac{i}{\sum_{j=1}^{N} j} )
            ```

        DECAY_EXPONENTIAL_VOTING (int):
            Weights each classifier's vote based on the recency of its wave using an exponential decay.
            Weight formula:

            ```math
            ( w_i = \\frac{e^{i}}{\sum_{j=1}^{N} e^{j}} )
            ```

        CV_BASED_VOTING (int):
            Weights each classifier based on its cross-validation accuracy on the training data.
            Weight formula:

            ```math
            ( w_i = \\frac{A_i}{\sum_{j=1}^{N} A_j} )
            ```

        STACKING (int):
            Stacking ensemble strategy uses a meta-learner to combine predictions of base classifiers.
            The meta-learner is trained on meta-features formed from the base classifiers' predictions.
            This approach is suitable when the cardinality of meta-features is smaller than the original feature set.

            In stacking, for each wave \\( i \\) (\\( i \\in \\{1, 2, \\ldots, N\\} \\)), a base classifier \\( C_i \\)
            is trained on \\( (X_i, T_N) \\). The prediction from \\( C_i \\) is denoted as \\( V_i \\), forming
            the meta-features \\( \\mathbf{V} = [V_1, V_2, ..., V_N] \\). The meta-learner \\( M \\) is then trained
            on \\( (\\mathbf{V}, T_N) \\), and for a new instance \\( x \\), the final prediction is
            \\( P(x) = M(\\mathbf{V}(x)) \\).

    """

    MAJORITY_VOTING = auto()
    DECAY_LINEAR_VOTING = auto()
    DECAY_EXPONENTIAL_VOTING = auto()
    CV_BASED_VOTING = auto()
    STACKING = auto()


class LongitudinalVotingClassifier(CustomClassifierMixinEstimator):
    """
    Longitudinal Voting Classifier for ensemble learning on longitudinal data.

    The Longitudinal Voting Classifier is a versatile ensemble method designed to handle the unique challenges posed by
    longitudinal data. It leverages different voting strategies to combine predictions from multiple base estimators,
    enhancing predictive performance. The base estimators are individually trained, and their predictions are aggregated
    based on the chosen voting strategy to generate the final prediction.

    !!! warning "When to Use?"
        This classifier is primarily used when the "SepWav" (Separate Waves) strategy is employed. However, it can also
        be applied with only longitudinal-based estimators that do not follow the SepWav approach if desired.

    !!! info "SepWav (Separate Waves) Strategy"

        The SepWav strategy involves considering each wave's features and the class variable as a separate dataset,
        then learning a classifier for each dataset. The class labels predicted by these classifiers are combined into
        a final predicted class label. This combination can be achieved using various approaches: simple majority voting,
        weighted voting with weights decaying linearly or exponentially for older waves, weights optimized by
        cross-validation on the training set (current class), and stacking methods that use the classifiers' predicted
        labels as input for learning a meta-classifier (see LongitudinalStacking).

    !!! info "Wrapper Around Sklearn VotingClassifier"

        This class wraps the `sklearn` VotingClassifier, offering a familiar interface while incorporating enhancements
        for longitudinal data.

    Args:
        voting (LongitudinalEnsemblingStrategy, default=LongitudinalEnsemblingStrategy.MAJORITY_VOTING):
            The voting strategy to be used for the ensemble. Refer to the LongitudinalEnsemblingStrategy enum.
        estimators (List[CustomClassifierMixinEstimator]):
            A list of classifiers for the ensemble. Note that the classifiers need to be trained before being passed to
            the LongitudinalVotingClassifier.
        extract_wave (Callable, optional):
            A function to extract specific wave data for training. Defaults to None.
        n_jobs (int, default=1):
            The number of jobs to run in parallel.

    Attributes:
        clf_ensemble (LongitudinalCustomVoting):
            The underlying custom voting classifier instance.

    Raises:
        ValueError: If no estimators are provided or if an invalid voting strategy is specified.
        NotFittedError: If attempting to predict or predict_proba before fitting the model.

    Examples:
        !!! example "Basic Usage with Dummy Longitudinal Data"
            ```python
            from scikit_longitudinal.estimators.ensemble.longitudinal_voting import (
                LongitudinalVotingClassifier,
                LongitudinalEnsemblingStrategy
            )
            from sklearn.ensemble import RandomForestClassifier
            from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoRandomForestClassifier
            import numpy as np

            # Dummy data
            X = np.array([[0, 1, 0, 1, 45, 1], [1, 1, 1, 1, 50, 0], [0, 0, 0, 0, 55, 1]])
            y = np.array([0, 1, 0])
            features_group = [[0, 1], [2, 3]]

            # Train estimators
            rf = RandomForestClassifier().fit(X, y)
            lexico_rf = LexicoRandomForestClassifier(features_group=features_group).fit(X, y)

            # Create and fit the voting classifier
            clf = LongitudinalVotingClassifier(
                voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
                estimators=[('rf', rf), ('lexico_rf', lexico_rf)],
            )
            clf.fit(X, y)
            y_pred = clf.predict(X)
            print(f"Predictions: {y_pred}")
            ```

        !!! example "Using Cross-Validation-Based Weighted Voting"
            ```python
            clf = LongitudinalVotingClassifier(
                voting=LongitudinalEnsemblingStrategy.CV_BASED_VOTING,
                estimators=[('rf', rf), ('lexico_rf', lexico_rf)],
            )
            clf.fit(X, y)
            y_pred = clf.predict(X)
            print(f"Predictions: {y_pred}")
            ```

    Notes:
        - **References**:

          - Ribeiro, C. and Freitas, A.A., 2019. "A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets." *3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL)*, held as part of IJCAI-2019.
    """

    def __init__(
        self,
        estimators: List[CustomClassifierMixinEstimator],
        voting: LongitudinalEnsemblingStrategy = LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
        extract_wave: Callable = None,
        n_jobs: int = 1,
    ) -> None:
        self.estimators = estimators
        self.voting = voting
        self.extract_wave = extract_wave
        self.n_jobs = n_jobs
        self.clf_ensemble = None

    @property
    def classes_(self):
        """
        Property to access the classes of the fitted ensemble model.

        Returns:
            np.ndarray: The class labels.

        Raises:
            NotFittedError: If the model is not fitted yet.
        """
        if self.clf_ensemble is None:
            raise NotFittedError(
                "This LongitudinalVotingClassifier instance is not fitted yet. Call 'fit' with appropriate arguments."
            )
        return self.clf_ensemble.classes_

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "LongitudinalVotingClassifier":
        """
        Fit the ensemble model.

        Trains the ensemble based on the specified voting strategy.

        Args:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.

        Returns:
            LongitudinalVotingClassifier:
                The fitted ensemble model.

        Raises:
            ValueError: If no estimators are provided or if an invalid voting strategy is specified.
            NotFittedError: If attempting to predict or predict_proba before fitting the model.

        !!! tip "Estimator Training"
            Ensure all estimators are trained before passing them to the `LongitudinalVotingClassifier`.
        """
        if self.estimators:
            for _, estimator in self.estimators:
                check_is_fitted(estimator, msg="Estimators must be fitted before using this method.")
        else:
            raise ValueError("No estimators were provided.")

        if not isinstance(self.voting, LongitudinalEnsemblingStrategy):
            raise ValueError(
                f"Invalid ensemble strategy. It must be a value from {LongitudinalEnsemblingStrategy} enum."
            )

        strategy_method = {
            LongitudinalEnsemblingStrategy.MAJORITY_VOTING: self._fit_majority_voting,
            LongitudinalEnsemblingStrategy.DECAY_LINEAR_VOTING: self._fit_decay_linear_voting,
            LongitudinalEnsemblingStrategy.DECAY_EXPONENTIAL_VOTING: self._fit_decay_exponential_voting,
            LongitudinalEnsemblingStrategy.CV_BASED_VOTING: self._fit_cv_based_voting,
        }.get(self.voting)

        if strategy_method:
            strategy_method(X, y)
        else:
            raise ValueError(f"Invalid ensemble strategy: {self.voting}")

        return self

    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ensemble model.

        Generates predictions based on the aggregated votes of the base estimators.

        Args:
            X (np.ndarray):
                The test data.

        Returns:
            np.ndarray:
                The predicted values.

        Raises:
            NotFittedError: If attempting to predict before fitting the model.

        !!! tip "Tie-Breaking"
            In case of a tie, the prediction from the most recent wave is selected.
        """
        if self.clf_ensemble:
            return self.clf_ensemble.predict(X)
        raise NotFittedError("Ensemble model is not fitted yet.")

    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the ensemble model.

        Generates probability estimates by averaging the probabilities from the base estimators.

        Args:
            X (np.ndarray):
                The test data.

        Returns:
            np.ndarray:
                The predicted probabilities.

        Raises:
            NotFittedError: If attempting to predict before fitting the model.
        """
        if self.clf_ensemble:
            return self.clf_ensemble.predict_proba(X)
        raise NotFittedError("Ensemble model is not fitted yet.")

    def _extract_wave(self, X: np.ndarray, wave: int) -> np.ndarray:
        """
        Extract the data for the given wave.

        Uses the `extract_wave` function to retrieve specific wave data.

        Args:
            X (np.ndarray):
                The training data.
            wave (int):
                The wave number to extract.

        Returns:
            np.ndarray:
                The extracted data.
        """
        if self.extract_wave:
            return X[:, self.extract_wave(wave, extract_indices=True)[2]]
        return X

    def _fit_majority_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ensemble model using majority voting strategy.

        Each estimator's vote is equally weighted.

        Args:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.
        """
        self.clf_ensemble = LongitudinalCustomVoting(self.estimators, extract_wave=self.extract_wave)
        self.clf_ensemble.fit(X, y)

    def _fit_decay_linear_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ensemble model using linear decay weighted voting strategy.

        Weights are assigned linearly, with more recent waves having higher weights.

        Args:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.
        """
        weights = self._calculate_linear_decay_weights(len(self.estimators))
        self.clf_ensemble = LongitudinalCustomVoting(self.estimators, weights=weights, extract_wave=self.extract_wave)
        self.clf_ensemble.fit(X, y)

    def _fit_decay_exponential_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ensemble model using exponential decay weighted voting strategy.

        Weights are assigned exponentially, favoring more recent waves.

        Args:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.
        """
        weights = self._calculate_exponential_decay_weights(len(self.estimators))
        self.clf_ensemble = LongitudinalCustomVoting(self.estimators, weights=weights, extract_wave=self.extract_wave)
        self.clf_ensemble.fit(X, y)

    def _fit_cv_based_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ensemble model using cross-validation weighted voting strategy.

        Weights are based on each estimator's cross-validation accuracy.

        Args:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.
        """
        weights = self._calculate_cv_weights(X, y, k=5)
        self.clf_ensemble = LongitudinalCustomVoting(self.estimators, weights=weights, extract_wave=self.extract_wave)
        self.clf_ensemble.fit(X, y)

    def _calculate_cv_weights(self, X: np.ndarray, y: np.ndarray, k: int) -> List[float]:
        """
        Calculate the weights based on cross-validation accuracy.

        Args:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.
            k (int):
                The number of folds for cross-validation.

        Returns:
            List[float]: Weights for each estimator.
        """
        accuracies = [
            cross_val_score(estimator, self._extract_wave(X, int(wave_number.split("_")[-1])), y, cv=k).mean()
            for wave_number, estimator in self.estimators
        ]
        total_accuracy = sum(accuracies)
        return [acc / total_accuracy for acc in accuracies]

    @staticmethod
    def _calculate_linear_decay_weights(N: int) -> List[float]:
        """
        Calculate the weights based on linear decay.

        Args:
            N (int):
                The number of waves.

        Returns:
            List[float]: Linear decay weights.
        """
        return [i / sum(range(1, N + 1)) for i in range(1, N + 1)]

    @staticmethod
    def _calculate_exponential_decay_weights(N: int) -> List[float]:
        """
        Calculate the weights based on exponential decay.

        Args:
            N (int):
                The number of waves.

        Returns:
            List[float]: Exponential decay weights.
        """
        return [np.exp(i) / sum(np.exp(j) for j in range(1, N + 1)) for i in range(1, N + 1)]
