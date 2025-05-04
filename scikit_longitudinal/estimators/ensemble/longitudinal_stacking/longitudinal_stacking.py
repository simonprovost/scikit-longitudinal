# pylint: disable=W0222,R0801
from typing import List, Optional, Union

import numpy as np
from overrides import override
from sklearn.base import ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from scikit_longitudinal.templates import CustomClassifierMixinEstimator


class LongitudinalStackingClassifier(CustomClassifierMixinEstimator):
    """
    Longitudinal Stacking Classifier for ensemble learning on longitudinal data.

    The Longitudinal Stacking Classifier is a sophisticated ensemble method designed to handle the unique challenges posed
    by longitudinal data. It leverages a stacking approach where multiple base estimators are trained, and their predictions
    are used as input features for a meta-learner, which generates the final prediction. This method excels at capturing
    complex temporal patterns by learning from the combined strengths of diverse base models.

    !!! warning "When to Use?"
        This classifier is primarily used with the "SepWav" (Separate Waves) strategy but can also be applied with
        longitudinal-based estimators that do not follow the SepWav approach if preferred.

    !!! info "SepWav (Separate Waves) Strategy"
        The SepWav strategy involves training separate classifiers for each wave's features and the class variable.
        The predictions from these classifiers are then combined using stacking, where a meta-learner (e.g., Logistic
        Regression, Decision Tree, or Random Forest) learns to make the final prediction based on the base classifiers'
        outputs.

    !!! info "Wrapper Around Sklearn StackingClassifier"
        This class wraps the `sklearn` StackingClassifier, offering a familiar interface while incorporating enhancements
        for longitudinal data.

    Args:
        estimators (List[CustomClassifierMixinEstimator]):
            The base estimators for the ensemble. These must be pre-trained before passing to the classifier.
        meta_learner (Optional[Union[CustomClassifierMixinEstimator, ClassifierMixin]], default=LogisticRegression()):
            The meta-learner to be used in stacking. Can be any scikit-learn compliant classifier.
        n_jobs (int, default=1):
            The number of jobs to run in parallel for fitting base estimators.

    Attributes:
        clf_ensemble (StackingClassifier):
            The underlying scikit-learn StackingClassifier instance.

    Raises:
        ValueError: If no base estimators are provided or if the meta-learner is not suitable.
        NotFittedError: If attempting to predict or predict_proba before fitting the model or if base estimators are not fitted.

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
            y = np.array([0, 1, 0])
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

          - Ribeiro, C. and Freitas, A.A., 2019. "A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets." *3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL)*, held as part of IJCAI-2019.
    """

    def __init__(
        self,
        estimators: List[CustomClassifierMixinEstimator],
        meta_learner: Optional[Union[CustomClassifierMixinEstimator, ClassifierMixin]] = LogisticRegression(),
        n_jobs: int = 1,
    ) -> None:
        self.estimators = estimators
        self.meta_learner = meta_learner
        self.n_jobs = n_jobs
        self.clf_ensemble = None

    @property
    def classes_(self):
        if self.clf_ensemble is None:
            raise NotFittedError(
                "This LongitudinalStackingClassifier instance is not fitted yet. Call 'fit' with appropriate arguments."
            )
        return self.clf_ensemble.classes_

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "LongitudinalStackingClassifier":
        """
        Fit the ensemble model.

        Trains the stacking ensemble by combining predictions from pre-trained base estimators and fitting the meta-learner.

        Args:
            X (np.ndarray):
                The input data.
            y (np.ndarray):
                The target data.

        Returns:
            LongitudinalStackingClassifier: The fitted model.

        Raises:
            ValueError: If no base estimators are provided or if the meta-learner is not suitable.
            NotFittedError: If any base estimators are not fitted.

        !!! tip "Meta-Learner Selection"
            Choose a meta-learner that complements your base estimators. For example, use Logistic Regression for linear
            decision boundaries or a Decision Tree for more complex interactions.
        """
        if self.estimators:
            for _, estimator in self.estimators:
                check_is_fitted(estimator, msg="Estimators must be fitted before using this method.")
        else:
            raise ValueError("No estimators were provided.")

        if not hasattr(self.meta_learner, "fit") and not hasattr(self.meta_learner, "predict"):
            raise ValueError("The meta learner must be a classifier with a fit and predict scikit-compliant format.")

        self.clf_ensemble = StackingClassifier(
            estimators=self.estimators, final_estimator=self.meta_learner, n_jobs=self.n_jobs
        )

        self.clf_ensemble.fit(X, y)
        return self

    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ensemble model.

        Generates predictions by passing the base estimators' predictions to the meta-learner.

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

        Generates probability estimates from the meta-learner based on base estimators' predictions.

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
