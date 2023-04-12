# pylint: disable=W0621,

import numpy as np
import pytest
from scipy.io import loadmat

from scikit_longitudinal.preprocessing.feature_selection.cfs_per_group.cfs_per_group import (
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
        (
            i,
            i + 1,
            i + 2,
            i + 3,
            i + 4,
        )
        for i in range(0, 50, 5)
    ]
    return X, y, group_features


class TestCFSPerGroup:
    @pytest.mark.parametrize(
        "search_method,expected_features",
        [
            ("exhaustiveSearch", [1, 2, 4, 6]),
            ("greedySearch", [4, 2, 6, 1]),
            ("forwardBestFirstSearch", [4, 2, 6, 1, 9, 3, 7, 5, 8]),
            ("backwardBestFirstSearch", [0, 1, 3, 5, 6, 7, 8, 9]),
            ("bidirectionalSearch", [4, 2, 6, 1, 9, 3, 7, 5, 8, 0]),
        ],
    )
    def test_feature_selection_non_longitudinal(
        self, load_madelon_data_non_longitudinal, search_method, expected_features
    ):
        X, y = load_madelon_data_non_longitudinal
        estimator = CorrelationBasedFeatureSelectionPerGroup(
            search_method=search_method, consecutive_non_improving_subsets_limit=5
        )
        estimator.fit_transform(X, y)
        np.testing.assert_array_equal(estimator.selected_features_, expected_features)

    @pytest.mark.parametrize(
        "search_method,expected_features",
        [
            ("forwardBestFirstSearch", [19, 18, 5, 6, 27, 24, 10, 13, 22, 7, 1, 26, 17, 14]),
            (
                "backwardBestFirstSearch",
                [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            ),
            ("greedySearch", [19, 18, 5, 6, 27, 24, 10, 13, 22]),
        ],
    )
    def test_feature_selection_longitudinal(self, load_madelon_data_longitudinal, search_method, expected_features):
        X, y, group_features = load_madelon_data_longitudinal

        estimator = CorrelationBasedFeatureSelectionPerGroup(
            search_method="exhaustiveSearch",
            consecutive_non_improving_subsets_limit=5,
            group_features=group_features,
            cfs_longitudinal_outer_search_method=search_method,
            parallel=False,
        )
        estimator.fit_transform(X, y)
        np.testing.assert_array_equal(estimator.selected_features_, expected_features)
