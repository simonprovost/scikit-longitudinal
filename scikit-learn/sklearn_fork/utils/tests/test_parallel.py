import time

import joblib
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn_fork import config_context, get_config
from sklearn_fork.compose import make_column_transformer
from sklearn_fork.datasets import load_iris
from sklearn_fork.ensemble import RandomForestClassifier
from sklearn_fork.model_selection import GridSearchCV
from sklearn_fork.pipeline import make_pipeline
from sklearn_fork.preprocessing import StandardScaler

from sklearn_fork.utils.parallel import delayed, Parallel


def get_working_memory():
    return get_config()["working_memory"]


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("backend", ["loky", "threading", "multiprocessing"])
def test_configuration_passes_through_to_joblib(n_jobs, backend):
    # Tests that the global global configuration is passed to joblib jobs

    with config_context(working_memory=123):
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(get_working_memory)() for _ in range(2)
        )

    assert_array_equal(results, [123] * 2)


def test_parallel_delayed_warnings():
    """Informative warnings should be raised when mixing sklearn_fork and joblib API"""
    # We should issue a warning when one wants to use sklearn_fork.utils.fixes.Parallel
    # with joblib.delayed. The config will not be propagated to the workers.
    warn_msg = "`sklearn_fork.utils.parallel.Parallel` needs to be used in conjunction"
    with pytest.warns(UserWarning, match=warn_msg) as records:
        Parallel()(joblib.delayed(time.sleep)(0) for _ in range(10))
    assert len(records) == 10

    # We should issue a warning if one wants to use sklearn_fork.utils.fixes.delayed with
    # joblib.Parallel
    warn_msg = (
        "`sklearn_fork.utils.parallel.delayed` should be used with "
        "`sklearn_fork.utils.parallel.Parallel` to make it possible to propagate"
    )
    with pytest.warns(UserWarning, match=warn_msg) as records:
        joblib.Parallel()(delayed(time.sleep)(0) for _ in range(10))
    assert len(records) == 10


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_dispatch_config_parallel(n_jobs):
    """Check that we properly dispatch the configuration in parallel processing.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/25239
    """
    pd = pytest.importorskip("pandas")
    iris = load_iris(as_frame=True)

    class TransformerRequiredDataFrame(StandardScaler):
        def fit(self, X, y=None):
            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
            return super().fit(X, y)

        def transform(self, X, y=None):
            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
            return super().transform(X, y)

    dropper = make_column_transformer(
        ("drop", [0]),
        remainder="passthrough",
        n_jobs=n_jobs,
    )
    param_grid = {"randomforestclassifier__max_depth": [1, 2, 3]}
    search_cv = GridSearchCV(
        make_pipeline(
            dropper,
            TransformerRequiredDataFrame(),
            RandomForestClassifier(n_estimators=5, n_jobs=n_jobs),
        ),
        param_grid,
        cv=5,
        n_jobs=n_jobs,
        error_score="raise",  # this search should not fail
    )

    # make sure that `fit` would fail in case we don't request dataframe
    with pytest.raises(AssertionError, match="X should be a DataFrame"):
        search_cv.fit(iris.data, iris.target)

    with config_context(transform_output="pandas"):
        # we expect each intermediate steps to output a DataFrame
        search_cv.fit(iris.data, iris.target)

    assert not np.isnan(search_cv.cv_results_["mean_test_score"]).any()
