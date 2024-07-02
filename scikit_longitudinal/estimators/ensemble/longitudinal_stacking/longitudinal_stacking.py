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
    """A classifier for longitudinal data classification using a stacking ensemble approach.

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    This class implements a stacking ensemble method for longitudinal data, combining multiple (longitudinal-adapted)
    base estimators and a meta-learner. Each base estimator is fitted on the whole dataset, and their predictions are
    used as  input for the meta-learner, which makes the final prediction.

    Note: This class is a wrapper around the sklearn StackingClassifier, and it is not intended to be used directly
    given that nothing is longitudinal-adapted nor proofed for longitudinal data by a scientific paper. However,
    this class is currently used by the separate waves data longitudinal-adapted preparation technique.

    Attributes:
        estimators (List[CustomClassifierMixinEstimator]):
            The base estimators for the ensemble.
        meta_learner (Optional[Union[CustomClassifierMixinEstimator, ClassifierMixin]]):
            The meta-learner to be used in stacking.
        n_jobs (int):
            The number of jobs to run in parallel for fitting base estimators.
        clf_ensemble (StackingClassifier):
            The underlying sklearn StackingClassifier instance.

    Raises:
        ValueError: If no base estimators are provided or the meta learner is not suitable.
        NotFittedError: If attempting to predict or predict_proba before fitting the model.

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
        """Fits the ensemble model.

        Attributes:
            X (np.ndarray):
                The input data.
            y (np.ndarray):
                The target data.

        Returns:
            LongitudinalStackingClassifier: The fitted model.

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
        """Predicts the target data for the given input data.

        Attributes:
            X (np.ndarray):
                The input data.

        Returns:
            np.ndarray: The predicted target data.

        """
        if self.clf_ensemble:
            return self.clf_ensemble.predict(X)
        raise NotFittedError("Ensemble model is not fitted yet.")

    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target data probabilities for the given input data.

        Attributes:
            X (np.ndarray):
                The input data.

        Returns:
            np.ndarray: The predicted target data probabilities.

        """
        if self.clf_ensemble:
            return self.clf_ensemble.predict_proba(X)
        raise NotFittedError("Ensemble model is not fitted yet.")
