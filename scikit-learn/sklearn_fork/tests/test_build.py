import os
import pytest
import textwrap

from sklearn_fork import __version__
from sklearn_fork.utils._openmp_helpers import _openmp_parallelism_enabled


def test_openmp_parallelism_enabled():
    # Check that sklearn_fork is built with OpenMP-based parallelism enabled.
    # This test can be skipped by setting the environment variable
    # ``SKLEARN_SKIP_OPENMP_TEST``.
    if os.getenv("SKLEARN_SKIP_OPENMP_TEST"):
        pytest.skip("test explicitly skipped (SKLEARN_SKIP_OPENMP_TEST)")

    base_url = "dev" if __version__.endswith(".dev0") else "stable"
    err_msg = textwrap.dedent("""
        This test fails because scikit-learn has been built without OpenMP.
        This is not recommended since some estimators will run in sequential
        mode instead of leveraging thread-based parallelism.

        You can find instructions to build scikit-learn with OpenMP at this
        address:

            https://scikit-learn.org/{}/developers/advanced_installation.html

        You can skip this test by setting the environment variable
        SKLEARN_SKIP_OPENMP_TEST to any value.
        """).format(base_url)

    assert _openmp_parallelism_enabled(), err_msg
