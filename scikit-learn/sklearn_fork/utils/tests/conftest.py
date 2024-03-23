import pytest

import sklearn_fork


@pytest.fixture
def print_changed_only_false():
    sklearn_fork.set_config(print_changed_only=False)
    yield
    sklearn_fork.set_config(print_changed_only=True)  # reset to default
