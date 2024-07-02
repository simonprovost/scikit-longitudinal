import os
import platform
from typing import List


def check_pdm_variables() -> None:
    """
    Check for the presence of required PDM environment variables.

    This function checks whether the necessary PDM environment variables are set.
    Specifically, it looks for 'PDM_IN_ENV'. If any are missing, the program will exit.

    Note:
        Typically, 'PDM_IN_ENV' should be set to 'in-project'.
    """
    pdm_variables: List[str] = ["PDM_IN_ENV"]

    if missing_pdm_variables := [
        var
        for var in pdm_variables
        if os.environ.get(var) is None or os.environ.get(var) == "" or len(os.environ.get(var)) == 0
    ]:
        print(f"Missing PDM environment variables: {', '.join(missing_pdm_variables)}")
        print("Please set these variables before proceeding. Usually, `in-project` is the default.")
        exit(1)
