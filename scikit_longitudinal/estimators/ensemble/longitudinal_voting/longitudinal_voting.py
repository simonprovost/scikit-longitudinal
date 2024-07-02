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
    """An enum for the different longitudinal voting strategies.

    Attributes:
        MAJORITY_VOTING (int):
            Simple consensus voting where the most frequent prediction is selected.
        DECAY_LINEAR_VOTING (int):
            Weights each classifier's vote based on the recency of its wave, with two subtypes:
                - Linear Decay Voting: Weights are assigned linearly with more recent waves having higher weights.
                    Weight formula: \\( w_i = \frac{i}{\\sum_{j=1}^{N} j} \\)
                - Exponential Decay Voting: Weights are assigned exponentially, favoring more recent waves.
                    Weight formula: \\( w_i = \frac{e^{i}}{\\sum_{j=1}^{N} e^{j}} \\)
        CV_BASED_VOTING (int):
            Weights each classifier based on its cross-validation accuracy on the training data.
            Weight formula: \\( w_i = \frac{A_i}{\\sum_{j=1}^{N} A_j} \\)
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
    """A classifier for longitudinal data analysis using various voting strategies.

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    This class allows for ensemble learning using different longitudinal voting strategies. It aggregates
    predictions from multiple classifiers and determines the final output based on the chosen voting strategy.
    It incorporates a longitudinal custom voting classifier under the hood. This cutom voting, do proceed as
    how the voting classifier of sklearn does, but in case of a tie, the tie breaking criteria is based on the
    more recent wave's prediction. To get a more in-depth explanation of the voting classifier, see below.

    This class allows for ensemble learning using different longitudinal voting strategies. It aggregates
    predictions from multiple classifiers and determines the final output based on the chosen voting strategy.

    Voting Strategies:
    - Majority Voting:
        Simple consensus voting where the most frequent prediction is selected.
    - Decay-Based Weighted Voting:
        Weights each classifier's vote based on the recency of its wave, with two subtypes:
            - Linear Decay Voting: Weights are assigned linearly with more recent waves having higher weights.
              Weight formula: \\( w_i = \frac{i}{\\sum_{j=1}^{N} j} \\)
            - Exponential Decay Voting: Weights are assigned exponentially, favoring more recent waves.
              Weight formula: \\( w_i = \frac{e^{i}}{\\sum_{j=1}^{N} e^{j}} \\)
    - Cross-Validation-Based Weighted Voting: Weights each classifier based on its cross-validation
      accuracy on the training data. Weight formula: \\( w_i = \frac{A_i}{\\sum_{j=1}^{N} A_j} \\)

    Final Prediction Calculation:
    - The final ensemble prediction \\( P \\) is derived from the votes \\( \\{V_1, V_2, \\ldots, V_N\\} \\)
      and their corresponding weights.
    - Formula: \\( P = \text{argmax}_{c} \\sum_{i=1}^{N} w_i \times I(V_i = c) \\)

    Tie-Breaking:
    - In the case of a tie, the most recent wave's prediction is selected as the final prediction. Note that
    this is only applicable for predict and not predict_proba, given that predict_proba does take the average of votes,
    similarly as how sklearn's voting classifier does.

    Attributes:
        estimators (List[CustomClassifierMixinEstimator]):
            A list of classifiers for the ensemble.
        voting (LongitudinalEnsemblingStrategy):
            The voting strategy to be used for the ensemble. Refer to the LongitudinalEnsemblingStrategy enum.
        extract_wave (Callable):
            A function to extract specific wave data for training.
        n_jobs (int):
            The number of jobs to run in parallel, this means the number of training classifier to do in parallel.
            Defaults to 1.
        clf_ensemble (LongitudinalCustomVoting):
            The underlying custom voting classifier instance.

    Raises:
        ValueError: If no estimators are provided or if an invalid voting strategy is specified.
        NotFittedError: If attempting to predict or predict_proba before fitting the model.

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
        if self.clf_ensemble is None:
            raise NotFittedError(
                "This LongitudinalVotingClassifier instance is not fitted yet. Call 'fit' with appropriate arguments."
            )
        return self.clf_ensemble.classes_

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "LongitudinalVotingClassifier":
        """Fit the ensemble model.

        Attributes:
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
        """Predict using the ensemble model.

        Attributes:
            X (np.ndarray):
                The test data.

        Returns:
            np.ndarray:
                The predicted values.

        Raises:
            NotFittedError: If attempting to predict before fitting the model.

        """
        if self.clf_ensemble:
            return self.clf_ensemble.predict(X)
        raise NotFittedError("Ensemble model is not fitted yet.")

    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the ensemble model.

        Attributes:
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
        """Extract the data for the given wave.

        Attributes:
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
        """Fit the ensemble model using majority voting strategy.

        Refer to the enum LongitudinalEnsemblingStrategy for more information on the voting strategy.

        Attributes:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.

        """
        self.clf_ensemble = LongitudinalCustomVoting(self.estimators, extract_wave=self.extract_wave)
        self.clf_ensemble.fit(X, y)

    def _fit_decay_linear_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the ensemble model using linear decay weighted voting strategy.

        Refer to the enum LongitudinalEnsemblingStrategy for more information on the voting strategy.

        Attributes:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.

        """
        weights = self._calculate_linear_decay_weights(len(self.estimators))
        self.clf_ensemble = LongitudinalCustomVoting(self.estimators, weights=weights, extract_wave=self.extract_wave)
        self.clf_ensemble.fit(X, y)

    def _fit_decay_exponential_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the ensemble model using exponential decay weighted voting strategy.

        Refer to the enum LongitudinalEnsemblingStrategy for more information on the voting strategy.

        Attributes:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.

        """
        weights = self._calculate_exponential_decay_weights(len(self.estimators))
        self.clf_ensemble = LongitudinalCustomVoting(self.estimators, weights=weights, extract_wave=self.extract_wave)
        self.clf_ensemble.fit(X, y)

    def _fit_cv_based_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the ensemble model using cross-validation weighted voting strategy.

        Refer to the enum LongitudinalEnsemblingStrategy for more information on the voting strategy.

        Attributes:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.

        """
        weights = self._calculate_cv_weights(X, y, k=5)
        self.clf_ensemble = LongitudinalCustomVoting(self.estimators, weights=weights, extract_wave=self.extract_wave)
        self.clf_ensemble.fit(X, y)

    def _calculate_cv_weights(self, X: np.ndarray, y: np.ndarray, k: int) -> List[float]:
        """Calculate the weights based on cross-validation accuracy.

        Attributes:
            X (np.ndarray):
                The training data.
            y (np.ndarray):
                The target values.
            k (int):
                The number of folds for cross-validation.

        """
        accuracies = [
            cross_val_score(estimator, self._extract_wave(X, int(wave_number.split("_")[-1])), y, cv=k).mean()
            for wave_number, estimator in self.estimators
        ]
        total_accuracy = sum(accuracies)
        return [acc / total_accuracy for acc in accuracies]

    @staticmethod
    def _calculate_linear_decay_weights(N: int) -> List[float]:
        """Calculate the weights based on linear decay.

        Attributes:
            N (int):
                The number of waves.

        """
        return [i / sum(range(1, N + 1)) for i in range(1, N + 1)]

    @staticmethod
    def _calculate_exponential_decay_weights(N: int) -> List[float]:
        """Calculate the weights based on exponential decay.

        Attributes:
            N (int):
                The number of waves.

        """
        return [np.exp(i) / sum(np.exp(j) for j in range(1, N + 1)) for i in range(1, N + 1)]
