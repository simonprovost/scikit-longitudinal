from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class WideTpTData:
    """Container holding the wide-format representation expected by TpT splitters."""

    X: np.ndarray
    y: np.ndarray
    feature_groups: List[List[int]]
    feature_names: List[str]
    subject_ids: List[str]
    time_indices: np.ndarray
    feature_columns: Sequence[str]


def _validate_long_dataframe(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    duration_col: str,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and normalize a LONG-format longitudinal dataframe for TpT preprocessing.

    This helper checks that the structural columns are present (subject id, time,
    duration), determines which columns should be treated as covariates, and
    returns a reordered copy of the dataframe sorted by (id, time).

    The returned dataframe has columns ordered as::

        feature_columns + [id_col, time_col, duration_col]

    and is sorted with a stable sort (mergesort) to preserve determinism.

    Parameters
    ----------
    df : pandas.DataFrame
        Input LONG-format dataframe containing one row per subject/time observation.
    id_col : str
        Name of the subject identifier column.
    time_col : str
        Name of the observation time column.
    duration_col : str
        Name of the subject-specific duration / horizon column.
    feature_columns : sequence of str, optional
        Columns to use as covariates. If ``None``, all columns except
        ``id_col``, ``time_col`` and ``duration_col`` are used. Any accidental
        inclusion of those structural columns is removed.

    Returns
    -------
    ordered : pandas.DataFrame
        A reordered copy of ``df`` restricted to the selected feature columns
        plus the structural columns, sorted by ``(id_col, time_col)``.
    feature_columns : list of str
        The finalized list of covariate column names (in the order used in
        ``ordered``).

    Raises
    ------
    ValueError
        If any required structural column is missing, or if no covariate column
        can be determined after excluding ``id_col``, ``time_col`` and
        ``duration_col``.
    """
    
    missing = {id_col, time_col, duration_col} - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns for LONG-format input: " + ", ".join(sorted(missing))
        )

    disallowed = {id_col, time_col, duration_col}
    if feature_columns is None:
        feature_columns = [
            col
            for col in df.columns
            if col not in disallowed
        ]
    if not feature_columns:
        raise ValueError("No feature columns found after excluding id/time/duration columns.")
    feature_columns = [col for col in feature_columns if col not in disallowed]
    if not feature_columns:
        raise ValueError("After removing id/time/duration columns no feature columns remain.")

    ordered = (
        df[feature_columns + [id_col, time_col, duration_col]]
        .copy()
        .sort_values([id_col, time_col], kind="mergesort")
        .reset_index(drop=True)
    )
    return ordered, list(feature_columns)


def long_to_wide(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    duration_col: str,
    time_step: float = 1.0,
    max_horizon: Optional[float] = None,
    feature_columns: Optional[Sequence[str]] = None,
    target: Optional[pd.Series] = None,
) -> WideTpTData:
    """Convert a LONG-format dataframe into the wide matrix required by TpT splitters.

    Parameters
    ----------
    df:
        Longitudinal dataframe containing one row per subject/time observation.
    id_col, time_col, duration_col:
        Column names identifying the subject id, the observation time and the
        subject-specific censoring time (horizon).
    time_step:
        Temporal granularity used to discretise times into waves.
    max_horizon:
        Optional cap for the temporal horizon. If omitted, the maximum duration
        observed in ``df`` is used.
    feature_columns:
        Optional subset of columns to treat as features. When omitted, all
        columns except ``id_col``, ``time_col`` and ``duration_col`` are used.

    Returns
    -------
    WideTpTData
        Dataclass with the dense feature matrix, feature groups and metadata.
    """

    ordered, feature_columns = _validate_long_dataframe(
        df,
        id_col=id_col,
        time_col=time_col,
        duration_col=duration_col,
        feature_columns=feature_columns,
    )

    # Discretise times into integer wave indices.
    if time_step <= 0:
        raise ValueError("time_step must be strictly positive.")

    time_index = np.floor_divide(
        np.asarray(ordered[time_col], dtype=np.float64) + 1e-9,
        time_step,
    ).astype(np.int64)

    duration_index = np.floor_divide(
        np.asarray(ordered[duration_col], dtype=np.float64) + 1e-9,
        time_step,
    ).astype(np.int64)

    max_observed_wave = int(np.floor(time_index.max())) if len(time_index) else 0
    if max_horizon is not None:
        max_horizon_wave = int(np.floor(max_horizon / time_step))
    else:
        max_horizon_wave = max_observed_wave

    max_time_index = min(max_observed_wave, max_horizon_wave)
    if max_time_index < 0:
        raise ValueError("No valid time horizon could be determined from the input data.")

    duration_index = np.minimum(duration_index, max_time_index)
    if max_time_index < 0:
        raise ValueError("No valid time horizon could be determined from the input data.")

    n_waves = max_time_index + 1

    # Build subject order to ensure deterministic layout.
    subjects: List[str] = ordered[id_col].astype(str).unique().tolist()
    subject_to_row: Dict[str, int] = {subject: idx for idx, subject in enumerate(subjects)}
    n_subjects = len(subjects)

    n_features = len(feature_columns)
    X = np.full((n_subjects, n_features * n_waves), np.nan, dtype=np.float64)
    feature_names: List[str] = []
    feature_groups: List[List[int]] = []

    # Pre-compute column offsets per feature.
    for feat_idx, feature in enumerate(feature_columns):
        group: List[int] = []
        for wave in range(n_waves):
            column_index = feat_idx * n_waves + wave
            feature_names.append(f"{feature}_t{wave}")
            group.append(column_index)
        feature_groups.append(group)

    # Iterate over subjects and fill the matrix with forward-filled measurements
    # until the subject horizon. Beyond the horizon we keep NaN so that TpT can
    # route those samples to duration leaves.
    grouped = ordered.assign(_time_index=time_index, _duration_index=duration_index).groupby(id_col, sort=False)

    if target is not None:
        # Align target with the ordered dataframe to preserve subject order.
        target_aligned = target.loc[ordered.index] if isinstance(target, pd.Series) else pd.Series(target)
    else:
        target_aligned = None

    subject_targets = np.full(n_subjects, np.nan)

    for subject, frame in grouped:
        row_idx = subject_to_row[str(subject)]
        first_original_idx = frame.index[0]
        frame = frame.reset_index(drop=True)
        if target_aligned is not None:
            subject_targets[row_idx] = target_aligned.loc[first_original_idx]
        times = frame.pop("_time_index").to_numpy(dtype=np.int64)
        durations = frame.pop("_duration_index").to_numpy(dtype=np.int64)
        subject_duration = int(durations.max())
        entry_wave = int(times.min())
        initial_values = {
            feature: frame.at[0, feature]
            for feature in feature_columns
        }

        # Track latest available values for each feature.
        last_values = {feature: np.nan for feature in feature_columns}
        pointer = 0

        for wave in range(n_waves):
            while pointer < len(frame) and times[pointer] <= wave:
                for feature in feature_columns:
                    last_values[feature] = frame.at[pointer, feature]
                pointer += 1

            if wave > subject_duration:
                # Subject no longer observed: keep NaNs so that the splitter treats
                # them as terminated subjects.
                continue

            for feat_idx, feature in enumerate(feature_columns):
                value = last_values[feature]
                if np.isnan(value) and wave < entry_wave:
                    value = initial_values.get(feature, np.nan)
                if np.isnan(value):
                    continue
                X[row_idx, feat_idx * n_waves + wave] = value

    time_indices = np.arange(n_waves, dtype=np.int64)

    # Targets are not produced here; the caller is responsible for selecting one
    # label per subject.
    return WideTpTData(
        X=X,
        y=subject_targets if target_aligned is not None else np.empty(n_subjects, dtype=np.float64),
        feature_groups=feature_groups,
        feature_names=feature_names,
        subject_ids=subjects,
        time_indices=time_indices,
        feature_columns=feature_columns,
    )
