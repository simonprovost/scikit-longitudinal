# pylint: disable=R0801

from abc import ABC
from functools import wraps
from typing import Any, Callable

import numpy as np
from overrides import EnforceOverrides, final
from sklearn.utils.validation import check_array, check_X_y


class DataPreparationMixin(ABC, EnforceOverrides):
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
    def prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> "DataPreparationMixin":
        if y is None:
            return self._check_array_decorator(self._prepare_data)(X)

        return self._check_X_y_decorator(self._prepare_data)(X, y)

    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> "DataPreparationMixin":
        raise NotImplementedError("Subclasses should implement _prepare_data method")
