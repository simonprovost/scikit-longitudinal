from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation._longitudinal_reshape import PAD_VALUE


def make_long_frame() -> pd.DataFrame:
    """Three subjects, three waves, two longitudinal features, one static."""
    return pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "bp": [120.0, 122.0, 121.0, 130.0, 131.0, 128.0, 110.0, 112.0, 115.0],
            "chol": [5.0, 5.5, 6.0, 4.5, 4.7, 5.0, 6.0, 6.0, 6.5],
            "sex": ["M", "M", "M", "F", "F", "F", "F", "F", "F"],
        }
    )


def make_wide_frame() -> pd.DataFrame:
    """Wide-format mirror with two longitudinal attrs over two waves and one static."""
    return pd.DataFrame(
        {
            "bp_w1": [120.0, 130.0, 110.0],
            "bp_w2": [122.0, 131.0, 112.0],
            "chol_w1": [5.0, 4.5, 6.0],
            "chol_w2": [5.5, 5.0, 6.0],
            "sex": ["M", "F", "F"],
        },
        index=[1, 2, 3],
    )


def _dataset(df: pd.DataFrame) -> LongitudinalDataset:
    return LongitudinalDataset(file_path=None, data_frame=df)


# ---------------------------------------------------------------------------
# LongitudinalDataset.to_wide
# ---------------------------------------------------------------------------


class TestToWide:
    def test_pivots_and_updates_dataset_state(self):
        dataset = _dataset(make_long_frame())
        wide = dataset.to_wide(
            id_col="subject_id",
            time_col="time",
            longitudinal_columns=["bp", "chol"],
            static_columns=["sex"],
        )
        assert dataset.data is wide
        assert list(wide.columns)[:4] == ["sex", "bp_w0", "bp_w1", "bp_w2"]
        assert wide.loc[1, "bp_w0"] == 120.0
        assert wide.loc[2, "chol_w1"] == 4.7
        assert dataset.feature_groups() == [[1, 2, 3], [4, 5, 6]]
        assert dataset.non_longitudinal_features() == [0]

    def test_no_static_columns(self):
        wide = _dataset(make_long_frame()).to_wide(
            id_col="subject_id",
            time_col="time",
            longitudinal_columns=["bp", "chol"],
        )
        assert list(wide.columns) == [
            "bp_w0", "bp_w1", "bp_w2", "chol_w0", "chol_w1", "chol_w2"
        ]

    def test_handles_unsorted_long_input(self):
        df = make_long_frame().sample(frac=1.0, random_state=7).reset_index(drop=True)
        wide = _dataset(df).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp", "chol"], static_columns=["sex"],
        )
        # Values must follow (subject, wave) regardless of input row order.
        assert wide.loc[1, "bp_w0"] == 120.0
        assert wide.loc[3, "chol_w2"] == 6.5

    def test_subjects_with_disjoint_wave_sets(self):
        df = pd.DataFrame({
            "subject_id": [1, 1, 2, 2, 3],
            "time":       [0, 1, 1, 2, 2],
            "bp":         [120.0, 121.0, 130.0, 131.0, 110.0],
            "sex":        ["M", "M", "F", "F", "F"],
        })
        wide = _dataset(df).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp"], static_columns=["sex"],
        )
        assert list(wide.columns) == ["sex", "bp_w0", "bp_w1", "bp_w2"]
        assert wide.loc[1, "bp_w0"] == 120.0
        assert pd.isna(wide.loc[1, "bp_w2"])
        assert pd.isna(wide.loc[2, "bp_w0"])
        assert pd.isna(wide.loc[3, "bp_w0"]) and pd.isna(wide.loc[3, "bp_w1"])
        assert wide.loc[3, "bp_w2"] == 110.0

    def test_string_wave_labels_sort_lexicographically(self):
        df = make_long_frame().copy()
        df["time"] = df["time"].map({0: "wA", 1: "wB", 2: "wC"})
        wide = _dataset(df).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp"], static_columns=["sex"],
        )
        assert list(wide.columns) == ["sex", "bp_wwA", "bp_wwB", "bp_wwC"]
        assert wide.loc[1, "bp_wwA"] == 120.0

    def test_custom_wave_format(self):
        wide = _dataset(make_long_frame()).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp"], static_columns=["sex"],
            wave_format="{feature}@t={wave}",
        )
        assert "bp@t=0" in wide.columns and "bp@t=2" in wide.columns

    def test_categorical_longitudinal_values_preserved(self):
        df = pd.DataFrame({
            "subject_id": [1, 1, 2, 2],
            "time":       [0, 1, 0, 1],
            "smoke":      ["yes", "no", "no", "no"],
            "sex":        ["M", "M", "F", "F"],
        })
        wide = _dataset(df).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["smoke"], static_columns=["sex"],
        )
        assert wide.loc[1, "smoke_w0"] == "yes"
        assert wide.loc[2, "smoke_w1"] == "no"
        # Non-numeric column should not be coerced to float.
        assert wide["smoke_w0"].dtype == object

    def test_nan_longitudinal_value_preserved_not_treated_as_missing_wave(self):
        df = make_long_frame().copy()
        df.loc[(df["subject_id"] == 1) & (df["time"] == 1), "bp"] = np.nan
        wide = _dataset(df).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp", "chol"], static_columns=["sex"],
        )
        assert pd.isna(wide.loc[1, "bp_w1"])
        # chol still present at wave 1
        assert wide.loc[1, "chol_w1"] == 5.5

    def test_single_subject_single_wave_each_still_pivots(self):
        df = pd.DataFrame({
            "subject_id": [1, 2, 3],
            "time":       [0, 0, 0],
            "bp":         [120.0, 130.0, 110.0],
            "chol":       [5.0, 4.5, 6.0],
        })
        wide = _dataset(df).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp", "chol"],
        )
        assert list(wide.columns) == ["bp_w0", "chol_w0"]
        assert wide.loc[2, "bp_w0"] == 130.0

    def test_subject_with_all_nan_longitudinals(self):
        df = make_long_frame().copy()
        df.loc[df["subject_id"] == 2, ["bp", "chol"]] = np.nan
        wide = _dataset(df).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp", "chol"], static_columns=["sex"],
        )
        assert wide.loc[2, ["bp_w0", "bp_w1", "bp_w2"]].isna().all()
        assert wide.loc[2, "sex"] == "F"

    def test_multiple_static_columns(self):
        df = make_long_frame().copy()
        df["country"] = "UK"
        wide = _dataset(df).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp"], static_columns=["sex", "country"],
        )
        assert list(wide.columns)[:2] == ["sex", "country"]
        assert wide.loc[1, "country"] == "UK"

    def test_writes_csv_when_output_path_given(self, tmp_path):
        out = tmp_path / "wide.csv"
        _dataset(make_long_frame()).to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp", "chol"], static_columns=["sex"],
            output_path=out,
        )
        assert out.exists()
        reloaded = pd.read_csv(out)
        assert "bp_w0" in reloaded.columns and "sex" in reloaded.columns

    # ---- error paths -------------------------------------------------------

    def test_rejects_missing_id_col(self):
        with pytest.raises(ValueError, match="id_col 'pid' not in dataframe"):
            _dataset(make_long_frame()).to_wide(
                id_col="pid", time_col="time", longitudinal_columns=["bp"],
            )

    def test_rejects_missing_time_col(self):
        with pytest.raises(ValueError, match="time_col 'wave' not in dataframe"):
            _dataset(make_long_frame()).to_wide(
                id_col="subject_id", time_col="wave", longitudinal_columns=["bp"],
            )

    def test_rejects_duplicate_id_time_rows(self):
        df = pd.concat([make_long_frame(), make_long_frame().iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate"):
            _dataset(df).to_wide(
                id_col="subject_id", time_col="time",
                longitudinal_columns=["bp", "chol"], static_columns=["sex"],
            )

    def test_rejects_static_varying_within_subject(self):
        df = make_long_frame()
        df.loc[0, "sex"] = "F"
        with pytest.raises(ValueError, match="Static columns vary"):
            _dataset(df).to_wide(
                id_col="subject_id", time_col="time",
                longitudinal_columns=["bp", "chol"], static_columns=["sex"],
            )

    def test_rejects_unknown_value_column(self):
        with pytest.raises(ValueError, match="Column 'mystery' not in dataframe"):
            _dataset(make_long_frame()).to_wide(
                id_col="subject_id", time_col="time",
                longitudinal_columns=["mystery"],
            )

    def test_rejects_unknown_static_column(self):
        with pytest.raises(ValueError, match="Column 'ghost' not in dataframe"):
            _dataset(make_long_frame()).to_wide(
                id_col="subject_id", time_col="time",
                longitudinal_columns=["bp"], static_columns=["ghost"],
            )

    def test_rejects_overlap_between_value_and_static(self):
        with pytest.raises(ValueError, match="listed more than once"):
            _dataset(make_long_frame()).to_wide(
                id_col="subject_id", time_col="time",
                longitudinal_columns=["bp"], static_columns=["bp"],
            )

    def test_rejects_id_col_as_longitudinal(self):
        with pytest.raises(ValueError, match="cannot be a value/static column"):
            _dataset(make_long_frame()).to_wide(
                id_col="subject_id", time_col="time",
                longitudinal_columns=["subject_id"],
            )

    def test_rejects_time_col_as_longitudinal(self):
        with pytest.raises(ValueError, match="cannot be a value/static column"):
            _dataset(make_long_frame()).to_wide(
                id_col="subject_id", time_col="time",
                longitudinal_columns=["time"],
            )

    def test_rejects_id_col_as_static(self):
        with pytest.raises(ValueError, match="cannot be a value/static column"):
            _dataset(make_long_frame()).to_wide(
                id_col="subject_id", time_col="time",
                longitudinal_columns=["bp"], static_columns=["subject_id"],
            )

    def test_rejects_empty_longitudinal_columns(self):
        with pytest.raises(ValueError, match="longitudinal_columns must list"):
            _dataset(make_long_frame()).to_wide(
                id_col="subject_id", time_col="time", longitudinal_columns=[],
            )


# ---------------------------------------------------------------------------
# LongitudinalDataset.to_long
# ---------------------------------------------------------------------------


class TestToLong:
    def _setup(self) -> LongitudinalDataset:
        dataset = _dataset(make_wide_frame())
        dataset.setup_features_group([["bp_w1", "bp_w2"], ["chol_w1", "chol_w2"]])
        return dataset

    def test_pivots_and_updates_dataset_state(self):
        dataset = self._setup()
        long_df = dataset.to_long(
            feature_base_names=["bp", "chol"], id_col="pid", time_col="wave",
        )
        assert dataset.data is long_df
        assert dataset.feature_groups() is None
        assert dataset.non_longitudinal_features() is None
        assert set(long_df.columns) == {"pid", "wave", "bp", "chol", "sex"}
        assert long_df.shape[0] == 6
        first = long_df[(long_df["pid"] == 1) & (long_df["wave"] == 1)].iloc[0]
        assert first["bp"] == 120.0 and first["sex"] == "M"

    def test_default_feature_base_names(self):
        dataset = self._setup()
        long_df = dataset.to_long(id_col="pid", time_col="wave")
        assert {"feature_0", "feature_1"}.issubset(long_df.columns)

    def test_keep_static_false_drops_static_columns(self):
        dataset = self._setup()
        long_df = dataset.to_long(
            feature_base_names=["bp", "chol"], id_col="pid",
            time_col="wave", keep_static=False,
        )
        assert "sex" not in long_df.columns
        assert set(long_df.columns) == {"pid", "wave", "bp", "chol"}

    def test_uneven_wave_counts_pad_with_nan(self):
        wide = make_wide_frame()
        wide["chol_w3"] = [np.nan, 5.5, 6.5]
        dataset = _dataset(wide)
        dataset.setup_features_group(
            [["bp_w1", "bp_w2"], ["chol_w1", "chol_w2", "chol_w3"]]
        )
        long_df = dataset.to_long(
            feature_base_names=["bp", "chol"], id_col="pid", time_col="wave",
        )
        # 3 subjects * 3 wave slots = 9 rows
        assert long_df.shape[0] == 9
        wave3 = long_df[long_df["wave"] == 3]
        assert wave3["bp"].isna().all()
        # Subject 1 had NaN at chol_w3 originally; subject 2 has 5.5
        assert pd.isna(wave3.loc[wave3["pid"] == 1, "chol"].iloc[0])
        assert wave3.loc[wave3["pid"] == 2, "chol"].iloc[0] == 5.5

    def test_pad_value_inside_group(self):
        # Build raw indices so we can poke a -1 in.
        wide = make_wide_frame()
        wide["chol_w3"] = [9.0, 8.0, 7.0]
        dataset = _dataset(wide)
        # bp has only 2 waves; pad slot for wave 3.
        cols = list(wide.columns)
        bp_group = [cols.index("bp_w1"), cols.index("bp_w2"), PAD_VALUE]
        chol_group = [cols.index("chol_w1"), cols.index("chol_w2"), cols.index("chol_w3")]
        dataset._feature_groups = [bp_group, chol_group]
        long_df = dataset.to_long(
            feature_base_names=["bp", "chol"], id_col="pid", time_col="wave",
        )
        wave3 = long_df[long_df["wave"] == 3]
        assert wave3["bp"].isna().all()  # padded
        assert wave3["chol"].tolist() == [9.0, 8.0, 7.0]

    def test_static_columns_inferred_from_unused_indices(self):
        wide = make_wide_frame().copy()
        wide["country"] = ["UK", "UK", "FR"]
        dataset = _dataset(wide)
        dataset.setup_features_group([["bp_w1", "bp_w2"], ["chol_w1", "chol_w2"]])
        long_df = dataset.to_long(
            feature_base_names=["bp", "chol"], id_col="pid", time_col="wave",
        )
        assert "country" in long_df.columns and "sex" in long_df.columns
        assert long_df.loc[long_df["pid"] == 3, "country"].iloc[0] == "FR"

    def test_non_default_index_is_carried_into_id_col(self):
        wide = make_wide_frame().copy()
        wide.index = ["alpha", "beta", "gamma"]
        dataset = _dataset(wide)
        dataset.setup_features_group([["bp_w1", "bp_w2"], ["chol_w1", "chol_w2"]])
        long_df = dataset.to_long(
            feature_base_names=["bp", "chol"], id_col="pid", time_col="wave",
        )
        assert set(long_df["pid"]) == {"alpha", "beta", "gamma"}

    def test_writes_csv_when_output_path_given(self, tmp_path):
        out = tmp_path / "long.csv"
        self._setup().to_long(
            feature_base_names=["bp", "chol"], id_col="pid",
            time_col="wave", output_path=out,
        )
        assert out.exists()
        reloaded = pd.read_csv(out)
        assert {"pid", "wave", "bp", "chol", "sex"}.issubset(reloaded.columns)

    # ---- error paths -------------------------------------------------------

    def test_rejects_when_features_group_missing(self):
        dataset = _dataset(make_wide_frame())
        with pytest.raises(ValueError, match="setup_features_group"):
            dataset.to_long()

    def test_rejects_wrong_length_feature_base_names(self):
        dataset = self._setup()
        with pytest.raises(ValueError, match="feature_base_names has 1 entries"):
            dataset.to_long(feature_base_names=["only_one"])

    def test_rejects_group_with_single_wave(self):
        dataset = _dataset(make_wide_frame())
        dataset._feature_groups = [[0]]
        with pytest.raises(ValueError, match="must list at least two waves"):
            dataset.to_long()

    def test_rejects_index_out_of_range(self):
        dataset = _dataset(make_wide_frame())
        dataset._feature_groups = [[0, 99]]
        with pytest.raises(ValueError, match="out of range"):
            dataset.to_long()

    def test_rejects_index_in_two_groups(self):
        dataset = _dataset(make_wide_frame())
        dataset._feature_groups = [[0, 1], [1, 2]]
        with pytest.raises(ValueError, match="appears in more than one group"):
            dataset.to_long()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_wide_to_long_then_long_to_wide_recovers_values(self):
        dataset = _dataset(make_wide_frame())
        dataset.setup_features_group([["bp_w1", "bp_w2"], ["chol_w1", "chol_w2"]])
        long_df = dataset.to_long(
            feature_base_names=["bp", "chol"], id_col="pid", time_col="wave",
        )
        wide = _dataset(long_df).to_wide(
            id_col="pid", time_col="wave",
            longitudinal_columns=["bp", "chol"], static_columns=["sex"],
        )
        assert wide.loc[1, "bp_w1"] == 120.0
        assert wide.loc[2, "chol_w2"] == 5.0
        assert list(wide["sex"]) == ["M", "F", "F"]

    def test_long_to_wide_then_wide_to_long_recovers_values(self):
        original = make_long_frame()
        dataset = _dataset(original)
        dataset.to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp", "chol"], static_columns=["sex"],
        )
        long_df = dataset.to_long(
            feature_base_names=["bp", "chol"], id_col="subject_id", time_col="time",
        )
        # Same number of rows; same totals.
        assert long_df.shape[0] == original.shape[0]
        # Wave labels were 0/1/2; to_long emits 1/2/3 (positional).
        # Compare per-subject sums (order-independent).
        for sid in original["subject_id"].unique():
            assert (
                long_df.loc[long_df["subject_id"] == sid, "bp"].sum()
                == pytest.approx(original.loc[original["subject_id"] == sid, "bp"].sum())
            )

    def test_round_trip_preserves_numeric_dtype(self):
        dataset = _dataset(make_long_frame())
        dataset.to_wide(
            id_col="subject_id", time_col="time",
            longitudinal_columns=["bp", "chol"], static_columns=["sex"],
        )
        for col in ["bp_w0", "bp_w1", "chol_w2"]:
            assert pd.api.types.is_numeric_dtype(dataset.data[col])
