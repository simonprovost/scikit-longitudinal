# pylint: disable=W0621,

import pandas as pd
import pytest
from scipy.io import loadmat

from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import (
    CorrelationBasedFeatureSelection,
    CorrelationBasedFeatureSelectionPerGroup,
)


@pytest.fixture(scope="module")
def load_madelon_data_non_longitudinal():
    mat = loadmat("scikit_longitudinal/tests/dummy_data/madelon.mat")
    X = mat["X"].astype(float)
    y = mat["Y"][:, 0]
    X = X[:, :10]
    return X, y


@pytest.fixture(scope="module")
def load_madelon_data_longitudinal():
    mat = loadmat("scikit_longitudinal/tests/dummy_data/madelon.mat")
    X = mat["X"].astype(float)
    y = mat["Y"][:, 0]
    X = X[:, :50]
    group_features = [
        [
            i,
            i + 1,
            i + 2,
            i + 3,
            i + 4,
        ]
        for i in range(0, 50, 5)
    ]
    return X, y, group_features


@pytest.fixture(scope="module")
def load_madelon_data_mixed():
    mat = loadmat("scikit_longitudinal/tests/dummy_data/madelon.mat")
    X = mat["X"].astype(float)
    y = mat["Y"][:, 0]
    X = X[:, :12]
    group_features = [
        [
            i,
            i + 1,
            i + 2,
            i + 3,
            i + 4,
        ]
        for i in range(0, 10, 5)
    ]
    return X, y, group_features


class TestCorrelationBasedFeatureSelectionPerGroup:
    @pytest.fixture(scope="module")
    def cfs_non_longitudinal(self):
        return CorrelationBasedFeatureSelection(search_method="greedySearch")

    @pytest.fixture(scope="module", params=[True, False])
    def cfs_longitudinal(self, load_madelon_data_longitudinal, request):
        _, _, group_features = load_madelon_data_longitudinal
        return CorrelationBasedFeatureSelectionPerGroup(
            search_method="greedySearch",
            features_group=group_features,
            parallel=request.param,
            outer_search_method="greedySearch",
            inner_search_method="greedySearch",
            version=1,
        )

    def test_fit_transform_non_longitudinal(self, cfs_non_longitudinal, load_madelon_data_non_longitudinal):
        X, y = load_madelon_data_non_longitudinal
        X_transformed = cfs_non_longitudinal.fit_transform(X, y)
        assert X_transformed.shape[1] == len(cfs_non_longitudinal.selected_features_)

    def test_fit_transform_longitudinal(self, cfs_longitudinal, load_madelon_data_longitudinal):
        X, y, group_features = load_madelon_data_longitudinal
        X_transformed = cfs_longitudinal.fit_transform(X, y)
        data = cfs_longitudinal.apply_selected_features_and_rename(
            pd.DataFrame(X_transformed), cfs_longitudinal.selected_features_
        )
        expected_shape = (cfs_longitudinal.selected_features_ or []) + (
            cfs_longitudinal.non_longitudinal_features or []
        )
        assert data.shape[1] == len(expected_shape)

    @pytest.mark.parametrize("search_method", ["greedySearch", "exhaustiveSearch"])
    def test_search_methods(self, search_method, load_madelon_data_non_longitudinal):
        X, y = load_madelon_data_non_longitudinal
        cfs = CorrelationBasedFeatureSelection(search_method=search_method)
        cfs.fit(X, y)
        assert len(cfs.selected_features_) > 0

    def test_apply_selected_features_and_rename_with_non_longitudinal_features(self, load_madelon_data_mixed):
        X, y, group_features = load_madelon_data_mixed
        df = pd.DataFrame(X, columns=[f"feature_{i}_w1" if i < 10 else f"gender_{i}" for i in range(X.shape[1])])
        df.rename(columns={"gender_10": "sex_w10", "gender_11": "gender_w11"}, inplace=True)
        selected_features = [0, 2, 4, 6, 8, 10, 11]

        cfs = CorrelationBasedFeatureSelectionPerGroup(
            non_longitudinal_features=[10, 11], search_method="greedySearch", features_group=group_features, version=1
        )
        cfs.fit(X, y)

        df_transformed = cfs.apply_selected_features_and_rename(df, selected_features)

        expected_columns = [f"feature_{i}_wave1" for i in [0, 2, 4, 6, 8]] + ["sex_wave10", "gender_wave11"]
        assert list(df_transformed.columns) == expected_columns

    def test_invalid_search_method(self, load_madelon_data_non_longitudinal):
        X, y = load_madelon_data_non_longitudinal
        with pytest.raises(AssertionError, match="search_method must be: 'exhaustiveSearch', or 'greedySearch'"):
            cfs = CorrelationBasedFeatureSelection(search_method="invalidMethod")
            cfs.fit(X, y)

    def test_single_feature_data(self, load_madelon_data_non_longitudinal):
        X, y = load_madelon_data_non_longitudinal
        X = X[:, :1]
        cfs = CorrelationBasedFeatureSelection(search_method="greedySearch")
        cfs.fit(X, y)
        assert len(cfs.selected_features_) == 1
