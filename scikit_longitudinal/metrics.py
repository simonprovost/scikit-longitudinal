# flake8: noqa
# pylint: skip-file

from sklearn.metrics import precision_recall_curve, auc
from typing import List
from functools import wraps


def validate_inputs(func):
    @wraps(func)
    def wrapper(y_true: List[int], y_score: List[float]):
        if len(y_true) != len(y_score):
            raise ValueError('Length of y_true and y_score must be the same.')
        if not all(isinstance(i, (int, float)) for i in y_score):
            raise ValueError('y_score should only contain numerical values.')
        if not all(isinstance(i, int) for i in y_true):
            raise ValueError('y_true should only contain integer values.')
        if any(i not in [0, 1] for i in y_true):
            raise ValueError('y_true should only contain binary values (0 and 1).')
        return func(y_true, y_score)

    return wrapper


@validate_inputs
def auprc_score(y_true: List[int], y_score: List[float]) -> float:
    """
    Calculate the Area Under the Precision-Recall Curve (AUPRC).

    Args:
        y_true (List[int]): Ground truth (correct) target values.
        y_score (List[float]): Estimated probabilities or decision function.

    Returns:
        float: Area under the precision-recall curve.

    Raises:
        ValueError: If `y_true` and `y_score` have different lengths.
        ValueError: If `y_score` contains non-numerical values.
        ValueError: If `y_true` contains non-integer values.
        ValueError: If `y_true` contains non-binary values (other than 0 and 1).

    Example:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_score = [0.1, 0.9, 0.8, 0.2, 0.65]
        >>> print(auprc_score(y_true, y_score))
        0.8300000000000001

    Notes:
        This implementation is based on the following article: https://towardsdatascience.com/
    the-wrong-and-right-way-to-approximate-area-under-precision-recall-curve-auprc-8fd9ca409064
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)
