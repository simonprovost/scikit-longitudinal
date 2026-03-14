from functools import wraps
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_consistent_length


def _binary_auprc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def _score_binary_target(
    y_true: np.ndarray, y_score: np.ndarray, labels: np.ndarray
) -> float:
    if labels.shape[0] != 2:
        raise ValueError("Binary targets require exactly two class labels.")

    positive_label = labels[1]
    y_true_binary = (np.asarray(y_true) == positive_label).astype(int)

    if y_score.ndim == 2:
        if y_score.shape[1] != 2:
            raise ValueError(
                "Binary targets require a 2-column score array when y_score is 2-dimensional."
            )
        y_score = y_score[:, 1]

    return _binary_auprc_score(y_true_binary, y_score)


def _binarize_nonbinary_target(
    y_true: np.ndarray, y_score: np.ndarray, y_type: str, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if y_score.ndim != 2:
        raise ValueError(
            "Multiclass targets require a 2-dimensional score array [n_samples, n_classes]."
        )

    if y_type == "multiclass":
        if y_score.shape[1] != labels.shape[0]:
            raise ValueError(
                "The number of score columns must match the number of class labels."
            )
        return label_binarize(y_true, classes=labels), labels

    y_true_binarized = np.asarray(y_true)
    if y_true_binarized.shape != y_score.shape:
        raise ValueError(
            "For multilabel targets, y_true and y_score must have the same shape."
        )
    if labels.shape[0] != y_score.shape[1]:
        labels = np.arange(y_score.shape[1])
    return y_true_binarized, labels


def _average_class_scores(
    y_true_binarized: np.ndarray, y_score: np.ndarray, average: Optional[str]
) -> Union[float, np.ndarray]:
    if average == "micro":
        return _binary_auprc_score(y_true_binarized.ravel(), y_score.ravel())

    per_class_scores = np.asarray(
        [
            _binary_auprc_score(y_true_binarized[:, index], y_score[:, index])
            for index in range(y_score.shape[1])
        ],
        dtype=float,
    )

    if average is None:
        return per_class_scores
    if average == "macro":
        return float(np.mean(per_class_scores))

    support = np.sum(y_true_binarized, axis=0, dtype=float)
    if np.isclose(np.sum(support), 0.0):
        raise ValueError(
            "weighted AUPRC is undefined when every class has zero support."
        )
    return float(np.average(per_class_scores, weights=support))


def metrics_validate_inputs(func):
    @wraps(func)
    def wrapper(
        y_true: Union[List[int], pd.Series, np.ndarray],
        y_score: Union[List[float], np.ndarray],
        average: Optional[str] = "macro",
        labels: Optional[Sequence] = None,
    ):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if y_score.ndim == 0 or y_score.ndim > 2:
            raise ValueError(
                "y_score should have at most 2 dimensions [n_samples, n_classes]."
            )
        if not np.issubdtype(y_score.dtype, np.number):
            raise ValueError("y_score should only contain numerical values.")

        if average not in {"micro", "macro", "weighted", None}:
            raise ValueError(
                "average must be one of {'micro', 'macro', 'weighted', None}."
            )

        if labels is not None and not isinstance(labels, Iterable):
            raise ValueError(
                "labels must be an iterable of class labels when provided."
            )

        check_consistent_length(y_true, y_score)
        return func(y_true, y_score.astype(float), average=average, labels=labels)

    return wrapper


@metrics_validate_inputs
def auprc_score(
    y_true: Union[List[int], pd.Series, np.ndarray],
    y_score: np.ndarray,
    average: Optional[str] = "macro",
    labels: Optional[Sequence] = None,
) -> Union[float, np.ndarray]:
    """Calculate the interpolated Area Under the Precision-Recall Curve (AUPRC).

    This metric computes the trapezoidal area under the precision-recall curve,
    i.e. the PR-AUC / AUPRC. It is intentionally distinct from scikit-learn's
    average precision (AP), which uses step-wise interpolation.

    Args:
        y_true (Union[List[int], pd.Series, np.ndarray]): Ground truth target values.
        y_score (np.ndarray): Estimated scores. Use shape ``(n_samples,)`` or
            ``(n_samples, 2)`` for binary targets, and ``(n_samples, n_classes)``
            for multiclass targets.
        average (Optional[str]): Averaging strategy for multiclass targets.
            Supported values are ``'micro'``, ``'macro'``, ``'weighted'``, or ``None``.
        labels (Optional[Sequence]): Explicit class order for multiclass or
            non-binary labels. When omitted, the class order is inferred from ``y_true``.

    Returns:
        Union[float, np.ndarray]: A scalar AUPRC or a per-class AUPRC array when
        ``average=None`` for multiclass targets.

    Raises:
        ValueError: If `y_true` and `y_score` have different lengths.
        ValueError: If `y_score` contains non-numerical values.
        ValueError: If `y_score` does not match the target cardinality.

    """
    y_type = type_of_target(y_true)
    labels = np.asarray(labels if labels is not None else unique_labels(y_true))

    if y_type == "binary":
        return _score_binary_target(y_true, y_score, labels)

    if y_type not in {"multiclass", "multilabel-indicator"}:
        raise ValueError(f"Unsupported target type for auprc_score: {y_type}.")

    y_true_binarized, _ = _binarize_nonbinary_target(y_true, y_score, y_type, labels)
    return _average_class_scores(y_true_binarized, y_score, average)
