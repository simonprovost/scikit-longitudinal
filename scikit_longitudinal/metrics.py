from functools import wraps
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target


def _binary_uninterpolated_average_precision(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def metrics_validate_inputs(func):
    @wraps(func)
    def wrapper(y_true: Union[List[int], pd.Series], y_score: np.ndarray):
        if not isinstance(y_score, np.ndarray):
            raise ValueError("y_score should be a numpy array.")
        if y_score.ndim > 2:
            raise ValueError("y_score should have at most 2 dimensions [n_samples, n_classes].")
        if not issubclass(y_score.dtype.type, np.floating):
            raise ValueError("y_score should only contain floating point numbers.")

        # Checking for binary class
        if len(y_true) > 0:
            first_element = y_true.iloc[0] if isinstance(y_true, pd.Series) else y_true[0]
            if (
                isinstance(y_true, (list, np.ndarray))
                and isinstance(first_element, np.ndarray)
                and len(first_element) == 2
            ):
                y_true = np.array([np.argmax(i) for i in y_true])

        # Validating y_true
        if not isinstance(y_true, (list, pd.Series, np.ndarray)) or not all(
            isinstance(i, (int, np.int64)) for i in y_true
        ):
            raise ValueError("y_true should be a list or a pandas Series, or numpy ndarray of integers.")
        if any(i not in [0, 1] for i in y_true):
            raise ValueError(
                "y_true should only contain binary values (0 and 1). Multi Class is not yet tested"
                "then not supported even though a partial implementation is present."
            )
        return func(y_true, y_score)

    return wrapper


@metrics_validate_inputs
def auprc_score(
    y_true: Union[List[int], pd.Series], y_score: np.ndarray, average: Optional[str] = "macro"
) -> Union[float, List[float]]:
    """Calculate the Area Under the Precision-Recall Curve (AUPRC).

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    Args:
        y_true (Union[List[int], pd.Series]): Ground truth (correct) target values.
        y_score (np.ndarray): Estimated probabilities or decision function.
        average (Optional[str]): {'micro', 'macro', 'samples', 'weighted'} or None

    Returns:
        Union[float, List[float]]: Area under the precision-recall curve.

    Raises:
        ValueError: If `y_true` and `y_score` have different lengths.
        ValueError: If `y_score` contains non-numerical values.
        ValueError: If `y_true` contains non-integer values.
        ValueError: If `y_true` contains non-binary values (other than 0 and 1).

    """
    y_type = type_of_target(y_true)
    if y_score.ndim == 2:  # pragma: no cover
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]  # select the scores for the positive class
        elif average == "micro":
            if y_type == "multiclass":
                y_true = label_binarize(y_true, classes=np.unique(y_true))
            return _binary_uninterpolated_average_precision(np.ravel(y_true), np.ravel(y_score))
        else:
            return _average_binary_score(_binary_uninterpolated_average_precision, y_true, y_score, average)
    return _binary_uninterpolated_average_precision(y_true, y_score)
