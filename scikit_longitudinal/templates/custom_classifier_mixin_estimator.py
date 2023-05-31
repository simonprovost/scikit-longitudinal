# pylint: disable=R0801

from functools import wraps
from typing import Any, Callable

import numpy as np
from overrides import EnforceOverrides, final
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y


class CustomClassifierMixinEstimator(BaseEstimator, ClassifierMixin, EnforceOverrides):
    """CustomClassifierMixinEstimator is a custom base class for scikit longitudinal estimators.

    A custom base class for scikit-learn estimators that automatically checks input data
    using the check_X_y and check_array functions from sklearn.utils.validation.

    Subclasses should implement the _fit and _predict methods.

    Methods:
        cannot be overriden - fit(X, y=None):
            Calls the _fit method, automatically checking X and y before calling.
        cannot be overriden - predict(X):
            Calls the _predict method, automatically checking X before calling.
        needs to be overriden - _fit(X, y=None):
            To be implemented by subclasses, contains the actual fitting logic.
        needs to be overriden -  _predict(X):
            To be implemented by subclasses, contains the actual prediction logic.

    """

    @staticmethod
    def _check_X_y_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(X: np.ndarray, y: np.ndarray, *args, **kwargs) -> Any:
            X, y = check_X_y(X, y)
            return func(X, y, *args, **kwargs)

        return wrapper

    @staticmethod
    def _check_array_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(X: np.ndarray, *args, **kwargs) -> Any:
            X = check_array(X)
            return func(X, *args, **kwargs)

        return wrapper

    @final
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "CustomClassifierMixinEstimator":
        if y is None:
            return self._check_array_decorator(self._fit)(X)
        return self._check_X_y_decorator(self._fit)(X, y)

    @final
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._check_array_decorator(self._predict)(X)

    @final
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._check_array_decorator(self._predict_proba)(X)

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "CustomClassifierMixinEstimator":
        raise NotImplementedError("Subclasses should implement _fit method")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement _predict method")

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement _predict_proba method")
