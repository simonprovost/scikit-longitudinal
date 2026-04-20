# pylint: disable=R0902,R0913,R0914,R0912,R0915

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PAD_VALUE = -1


def _format_wave_column(template: str, feature: str, wave: int) -> str:
    """Render the wide-format column name for a given feature/wave pair."""
    return template.format(feature=feature, wave=wave)


def _validate_wide_inputs(
    columns: Sequence[str],
    features_group: Sequence[Sequence[int]],
    non_longitudinal_features: Optional[Sequence[Union[int, str]]],
) -> Tuple[List[List[int]], List[int]]:
    """Normalise and validate `features_group` / `non_longitudinal_features` against `columns`."""
    if not features_group:
        raise ValueError("features_group must contain at least one group.")
    n_columns = len(columns)
    cleaned: List[List[int]] = []
    seen: set[int] = set()
    for grp_idx, group in enumerate(features_group):
        if not isinstance(group, (list, tuple)):
            raise ValueError(f"features_group[{grp_idx}] must be a list, got {type(group).__name__}.")
        without_pad = [idx for idx in group if idx != PAD_VALUE]
        if len(without_pad) < 2:
            raise ValueError(
                f"features_group[{grp_idx}] must list at least two waves; got {group}."
            )
        for idx in without_pad:
            if not isinstance(idx, (int, np.integer)):
                raise ValueError(
                    f"features_group[{grp_idx}] must contain integer indices; got {idx!r}."
                )
            if idx < 0 or idx >= n_columns:
                raise ValueError(
                    f"features_group[{grp_idx}] index {idx} out of range [0, {n_columns})."
                )
            if idx in seen:
                raise ValueError(
                    f"Index {idx} ('{columns[idx]}') appears in more than one group."
                )
            seen.add(idx)
        cleaned.append(list(group))

    if non_longitudinal_features is None:
        non_long_indices: List[int] = []
    else:
        non_long_indices = []
        for ref in non_longitudinal_features:
            if isinstance(ref, (int, np.integer)):
                idx = int(ref)
            elif isinstance(ref, str):
                if ref not in columns:
                    raise ValueError(f"non_longitudinal_features: column '{ref}' not found.")
                idx = list(columns).index(ref)
            else:
                raise ValueError(
                    f"non_longitudinal_features must contain int or str, got {type(ref).__name__}."
                )
            if idx < 0 or idx >= n_columns:
                raise ValueError(
                    f"non_longitudinal_features index {idx} out of range [0, {n_columns})."
                )
            if idx in seen:
                raise ValueError(
                    f"Column '{columns[idx]}' is declared in features_group and "
                    "non_longitudinal_features simultaneously."
                )
            non_long_indices.append(idx)
    return cleaned, non_long_indices


def _validate_long_inputs(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    longitudinal_columns: Sequence[str],
    static_columns: Sequence[str],
) -> None:
    """Validate that the long-format inputs reference real, non-overlapping columns."""
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not in dataframe.")
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in dataframe.")
    if not longitudinal_columns:
        raise ValueError("longitudinal_columns must list at least one longitudinal column.")
    roles = {id_col, time_col}
    seen: set[str] = set()
    for col in list(longitudinal_columns) + list(static_columns):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in dataframe.")
        if col in roles:
            raise ValueError(
                f"Column '{col}' cannot be a value/static column and id/time column."
            )
        if col in seen:
            raise ValueError(f"Column '{col}' listed more than once.")
        seen.add(col)


def long_to_wide(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    longitudinal_columns: Sequence[str],
    static_columns: Sequence[str] = (),
    wave_format: str = "{feature}_w{wave}",
) -> Tuple[pd.DataFrame, List[List[int]], List[int]]:
    """Pivot a long-format dataframe to wide format.

    Returns the wide dataframe, a `features_group` describing the wave layout,
    and the column indices of the static (non-longitudinal) columns.
    """
    _validate_long_inputs(df, id_col, time_col, longitudinal_columns, static_columns)

    longitudinal_columns = list(longitudinal_columns)
    static_columns = list(static_columns)

    duplicates = df.duplicated(subset=[id_col, time_col])
    if duplicates.any():
        offending = df.loc[duplicates, [id_col, time_col]].head(3).to_dict("records")
        raise ValueError(
            f"Duplicate (id, time) rows in long dataframe; got e.g. {offending}."
        )

    if static_columns:
        nunique = df.groupby(id_col)[static_columns].nunique(dropna=False)
        bad = nunique[(nunique > 1).any(axis=1)]
        if not bad.empty:
            raise ValueError(
                f"Static columns vary within a subject: {bad.index.tolist()[:3]}"
            )

    subject_ids = df[id_col].drop_duplicates().tolist()
    wave_labels = sorted(df[time_col].dropna().unique().tolist())
    n_subjects = len(subject_ids)
    n_waves = len(wave_labels)
    n_static = len(static_columns)
    wave_to_pos: Dict = {w: i for i, w in enumerate(wave_labels)}

    wide_columns: List[str] = []
    feature_groups: List[List[int]] = []
    for feat in longitudinal_columns:
        grp: List[int] = []
        for wave in wave_labels:
            grp.append(n_static + len(wide_columns))
            wide_columns.append(_format_wave_column(wave_format, feat, wave))
        feature_groups.append(grp)

    full_columns = list(static_columns) + wide_columns
    wide_matrix = np.full((n_subjects, len(full_columns)), np.nan, dtype=object)
    subject_to_row = {sid: i for i, sid in enumerate(subject_ids)}

    if static_columns:
        static_first = df.groupby(id_col, sort=False)[static_columns].first()
        for sid in subject_ids:
            wide_matrix[subject_to_row[sid], :n_static] = static_first.loc[sid].to_numpy()

    for sid, frame in df.groupby(id_col, sort=False):
        row = subject_to_row[sid]
        for _, observation in frame.iterrows():
            wave_pos = wave_to_pos[observation[time_col]]
            for feat_idx, col in enumerate(longitudinal_columns):
                j = n_static + feat_idx * n_waves + wave_pos
                wide_matrix[row, j] = observation[col]

    wide_df = pd.DataFrame(wide_matrix, columns=full_columns, index=subject_ids)
    wide_df.index.name = id_col
    for col in full_columns:
        try:
            wide_df[col] = pd.to_numeric(wide_df[col])
        except (ValueError, TypeError):
            pass

    return wide_df, feature_groups, list(range(n_static))


def wide_to_long(
    df: pd.DataFrame,
    *,
    features_group: Sequence[Sequence[int]],
    non_longitudinal_features: Optional[Sequence[Union[int, str]]] = None,
    feature_base_names: Optional[Sequence[str]] = None,
    id_col: str = "subject_id",
    time_col: str = "wave",
    keep_static: bool = True,
) -> pd.DataFrame:
    """Reshape a wide dataframe back to long format using `features_group`."""
    columns = list(df.columns)

    cleaned_groups, static_indices = _validate_wide_inputs(
        columns, features_group, non_longitudinal_features
    )

    if non_longitudinal_features is None:
        in_groups = {idx for grp in cleaned_groups for idx in grp if idx != PAD_VALUE}
        static_indices = [i for i in range(len(columns)) if i not in in_groups]
    static_cols = [columns[i] for i in static_indices]

    if feature_base_names is None:
        feature_base_names = [f"feature_{i}" for i in range(len(cleaned_groups))]
    elif len(feature_base_names) != len(cleaned_groups):
        raise ValueError(
            f"feature_base_names has {len(feature_base_names)} entries but features_group has "
            f"{len(cleaned_groups)} groups."
        )

    max_waves = max(len(g) for g in cleaned_groups)
    pieces: List[pd.DataFrame] = []
    indices = df.index.tolist()
    for wave_pos in range(max_waves):
        wave_frame = pd.DataFrame({id_col: indices, time_col: wave_pos + 1})
        wave_frame.index = df.index
        for base, group in zip(feature_base_names, cleaned_groups):
            if wave_pos < len(group) and group[wave_pos] != PAD_VALUE:
                wave_frame[base] = df.iloc[:, group[wave_pos]].values
            else:
                wave_frame[base] = np.nan
        if keep_static and static_cols:
            for col in static_cols:
                wave_frame[col] = df[col].values
        pieces.append(wave_frame)

    long_df = pd.concat(pieces, axis=0, ignore_index=True)
    long_df = long_df.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)

    return long_df


__all__ = ["long_to_wide", "wide_to_long"]
