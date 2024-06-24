import traceback
from functools import wraps
from typing import Any, Callable  # pragma: no cover

import numpy as np  # pragma: no cover
import pandas as pd  # pragma: no cover
import stopit
from rich import print  # pylint: disable=W0622


def handle_errors(f: Callable) -> Callable:
    """Decorator to handle errors and print tracebacks for functions. S.t fit only for the moment.

    This decorator catches exceptions, prints tracebacks, and then re-raises the exceptions.

    Args:
        f: The function to be wrapped.

    Returns:
        The wrapped function.

    Raises:
        stopit.utils.TimeoutException: If a timeout occurs during the function's execution.
        Exception: Any other exceptions that might occur during the function's execution.

    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        try:
            return f(*args, **kwargs)
        except stopit.utils.TimeoutException:
            print("Model training timed out, skipping...")
            raise
        except Exception as e:
            print(f"An error occurred in function {f.__name__}: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            raise

    return wrapper


def validate_input(f: Callable) -> Callable:
    """Decorator to validate the input passed to a function. S.t predict, predict_proba, transform, and fit.

    This decorator checks if the input data is not None and is one of the following types:
    - numpy array
    - pandas DataFrame
    - 2D list

    Additionally, if a target array `y` is passed, it checks if `y` is:
    - pandas Series
    - numpy array
    - 2D list

    Args:
        f:
            The function to be wrapped.

    Returns:
        The wrapped function with input validation.

    Raises:
        ValueError:
            - If the input data is None.
            - If the input data is not one of the expected types.
            - If the target data `y` is not one of the expected types.

    """

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        y = None
        list_args = list(args)

        X = list_args[1]

        if X is None:
            raise ValueError(f"No data was passed to {f.__name__}.")
        if not isinstance(X, (np.ndarray, pd.DataFrame, list)):
            raise ValueError("Input data must be a numpy array, pandas DataFrame, or 2D list.")

        if len(list_args) > 2:
            y = list_args[2]
            if y is not None and not isinstance(y, (pd.Series, np.ndarray, list)):
                raise ValueError("y must be a pandas Series, numpy array, 2D list, or not passing any target data")

        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        list_args[1] = X

        if len(list_args) > 2 and y is not None:
            if isinstance(y, list):
                y = np.array(y)
            elif isinstance(y, pd.Series):
                y = y.to_numpy()
            list_args[2] = y

        return f(*list_args, **kwargs)

    return wrapper
