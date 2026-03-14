# pylint: disable=W0222,R0801
from inspect import signature
from typing import Callable, List, Optional, Union

import numpy as np
from overrides import override
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from scikit_longitudinal.templates import CustomClassifierMixinEstimator


class _WaveAwareEstimator(BaseEstimator, ClassifierMixin):
    """Adapter that keeps a base estimator wave-specific inside sklearn stacking."""

    def __init__(self, estimator: ClassifierMixin, wave: int, extract_wave: Callable):
        self.estimator = estimator
        self.wave = wave
        self.extract_wave = extract_wave
        self.estimator_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        X, y = check_X_y(X, y)
        X_wave = self._extract_wave(X)
        self.estimator_ = clone(self.estimator)
        self.classes_ = unique_labels(y)

        fit_params = {}
        if (
            sample_weight is not None
            and "sample_weight" in signature(self.estimator_.fit).parameters
        ):
            fit_params["sample_weight"] = sample_weight

        self.estimator_.fit(X_wave, y, **fit_params)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        X = check_array(X)
        return self.estimator_.predict(self._extract_wave(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        X = check_array(X)
        probabilities = self.estimator_.predict_proba(self._extract_wave(X))
        estimator_classes = getattr(self.estimator_, "classes_", self.classes_)

        if np.array_equal(estimator_classes, self.classes_):
            return probabilities

        aligned_probabilities = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        for local_index, label in enumerate(estimator_classes):
            aligned_probabilities[:, class_to_index[label]] = probabilities[
                :, local_index
            ]
        return aligned_probabilities

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimator_")

    def _extract_wave(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.extract_wave(self.wave, extract_indices=True)[2]]


class LongitudinalStackingClassifier(CustomClassifierMixinEstimator):
    """
    Longitudinal Stacking Classifier for ensemble learning on longitudinal data.

    The Longitudinal Stacking Classifier is a sophisticated ensemble method designed to handle the unique challenges
    posed by longitudinal data. It leverages a stacking approach where multiple base estimators are trained, and their
    predicted class probabilities are used as input features for a meta-learner, which generates the final
    prediction. This method excels at capturing complex temporal patterns by learning from the combined strengths of
    diverse base models, and supports both binary and multiclass targets.

    !!! warning "When to Use?"
        This classifier is primarily used with the "SepWav" (Separate Waves) strategy but can also be applied with
        longitudinal-based estimators that do not follow the SepWav approach if preferred.

    !!! info "SepWav (Separate Waves) Strategy"
        The SepWav strategy involves training separate classifiers for each wave's features and the class variable.
        The class-probability outputs from these classifiers are then combined using stacking, where a meta-learner
        (e.g., Logistic Regression, Decision Tree, or Random Forest) learns to make the final prediction based on the
        base classifiers' outputs.

    !!! info "Wrapper Around Sklearn StackingClassifier"
        This class wraps the `sklearn` StackingClassifier, offering a familiar interface while incorporating
        enhancements for longitudinal data. As in scikit-learn, base estimators are cloned and refitted during
        stacking unless a prefit workflow is explicitly requested. When `extract_wave` is provided, those internal
        refits remain wave-specific.

    Args:
        estimators (List[CustomClassifierMixinEstimator]):
            The base estimators for the ensemble. These can be passed directly, or as estimators prepared by `SepWav`.
            Each estimator must implement `predict_proba`.
        meta_learner (Optional[Union[CustomClassifierMixinEstimator, ClassifierMixin]], default=LogisticRegression()):
            The meta-learner to be used in stacking. Can be any scikit-learn compliant classifier.
        n_jobs (int, default=1):
            The number of jobs to run in parallel for fitting base estimators.
        extract_wave (Callable, optional):
            Optional wave extractor used when estimators should remain wave-specific inside stacking, such as the
            `SepWav` workflow.

    Attributes:
        clf_ensemble (StackingClassifier):
            The underlying scikit-learn StackingClassifier instance.

    Raises:
        ValueError: If no base estimators are provided, if a base estimator does not implement `predict_proba`, or if
            the meta-learner is not suitable.
        NotFittedError: If attempting to predict or predict_proba before fitting the model.

    Examples:
        !!! example "Basic Usage with Dummy Longitudinal Data"

            ```python
            from scikit_longitudinal.estimators.ensemble.longitudinal_stacking import LongitudinalStackingClassifier
            from sklearn.ensemble import RandomForestClassifier
            from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoRandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            # Dummy data
            X = np.array([[0, 1, 0, 1, 45, 1], [1, 1, 1, 1, 50, 0], [0, 0, 0, 0, 55, 1]])
            y = np.array([0, 1, 2])
            features_group = [[0, 1], [2, 3]]

            # Train base estimators
            rf = RandomForestClassifier().fit(X, y)
            lexico_rf = LexicoRandomForestClassifier(features_group=features_group).fit(X, y)

            # Create and fit the stacking classifier
            clf = LongitudinalStackingClassifier(
                estimators=[('rf', rf), ('lexico_rf', lexico_rf)],
                meta_learner=LogisticRegression()
            )
            clf.fit(X, y)
            y_pred = clf.predict(X)
            print(f"Predictions: {y_pred}")
            ```

        !!! example "Using a Decision Tree as Meta-Learner with Parallel Processing"

            ```python
            from sklearn.tree import DecisionTreeClassifier
            clf = LongitudinalStackingClassifier(
                estimators=[('rf', rf), ('lexico_rf', lexico_rf)],
                meta_learner=DecisionTreeClassifier(),
                n_jobs=-1  # Use all available CPUs
            )
            clf.fit(X, y)
            y_pred = clf.predict(X)
            print(f"Predictions: {y_pred}")
            ```

    Notes:
        - **References**:

          - Ribeiro, C. and Freitas, A.A., 2019. "A mini-survey of supervised machine learning approaches for coping
            with ageing-related longitudinal datasets." *3rd Workshop on AI for Aging, Rehabilitation and Independent
            Assisted Living (ARIAL)*, held as part of IJCAI-2019.
    """

    def __init__(
        self,
        estimators: List[CustomClassifierMixinEstimator],
        meta_learner: Optional[
            Union[CustomClassifierMixinEstimator, ClassifierMixin]
        ] = LogisticRegression(),
        n_jobs: int = 1,
        extract_wave: Callable = None,
    ) -> None:
        self.estimators = estimators
        self.meta_learner = meta_learner
        self.n_jobs = n_jobs
        self.extract_wave = extract_wave
        self.clf_ensemble = None

    @property
    def classes_(self):
        if self.clf_ensemble is None:
            raise NotFittedError(
                "This LongitudinalStackingClassifier instance is not fitted yet. Call 'fit' with appropriate arguments."
            )
        return self.clf_ensemble.classes_

    @override
    def _fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LongitudinalStackingClassifier":
        """
        Fit the ensemble model.

        Trains the stacking ensemble by combining out-of-fold base-estimator probability predictions and fitting the
        meta-learner.

        Args:
            X (np.ndarray):
                The input data.
            y (np.ndarray):
                The target data.

        Returns:
            LongitudinalStackingClassifier: The fitted model.

        Raises:
            ValueError: If no base estimators are provided, if a base estimator does not implement `predict_proba`, or
                if the meta-learner is not suitable.

        !!! tip "Meta-Learner Selection"
            Choose a meta-learner that complements your base estimators. For example, use Logistic Regression for linear
            decision boundaries or a Decision Tree for more complex interactions.
        """
        if not self.estimators:
            raise ValueError("No estimators were provided.")

        if not hasattr(self.meta_learner, "fit") or not hasattr(
            self.meta_learner, "predict"
        ):
            raise ValueError(
                "The meta learner must be a classifier with a fit and predict scikit-compliant format."
            )
        if any(
            not hasattr(estimator, "predict_proba") for _, estimator in self.estimators
        ):
            raise ValueError(
                "All base estimators must implement predict_proba for LongitudinalStackingClassifier."
            )

        estimators = self.estimators
        stack_method = "predict_proba"
        if self.extract_wave is not None:
            estimators = [
                (
                    name,
                    _WaveAwareEstimator(
                        estimator=estimator, wave=wave, extract_wave=self.extract_wave
                    ),
                )
                for wave, (name, estimator) in enumerate(self.estimators)
            ]

        self.clf_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_learner,
            n_jobs=self.n_jobs,
            stack_method=stack_method,
        )

        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self.clf_ensemble.fit(X, y, **fit_params)
        return self

    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ensemble model.

        Generates predictions by passing stacked base-estimator probability outputs to the meta-learner.

        Args:
            X (np.ndarray):
                The input data.

        Returns:
            np.ndarray: The predicted target data.

        Raises:
            NotFittedError: If attempting to predict before fitting the model.
        """
        if self.clf_ensemble:
            return self.clf_ensemble.predict(X)
        raise NotFittedError("Ensemble model is not fitted yet.")

    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the ensemble model.

        Generates probability estimates from the meta-learner based on stacked base-estimator probability outputs.

        Args:
            X (np.ndarray):
                The input data.

        Returns:
            np.ndarray: The predicted target data probabilities.

        Raises:
            NotFittedError: If attempting to predict before fitting the model.

        !!! tip "Probability Calibration"
            If your meta-learner supports probability calibration (e.g., Logistic Regression), consider calibrating
            probabilities for better confidence estimates.
        """
        if self.clf_ensemble:
            return self.clf_ensemble.predict_proba(X)
        raise NotFittedError("Ensemble model is not fitted yet.")
