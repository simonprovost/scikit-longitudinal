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


def check_compiler_variables(system_platform: str) -> bool:
    """
    Check for the presence of required compiler environment variables based on the platform.

    Parameters:
        system_platform (str): Name of the operating system platform ('Darwin', 'Linux', etc.).

    Returns:
        bool: True if all required variables are set, False otherwise.

    Note:
        This function provides guidance for setting up compiler variables for macOS and Linux.
    """
    variables: List[str] = ["CC", "CXX", "CPPFLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS"]
    missing_variables: List[str] = []
    found_variables: List[str] = []

    for var in variables:
        if os.environ.get(var) is None:
            missing_variables.append(var)
        else:
            found_variables.append(var)

    if missing_variables:
        print(f"Missing compiler environment variables: {', '.join(missing_variables)}")
        print("Please set these variables before proceeding.")

        if system_platform == "Darwin":
            print(
                "For macOS, you can install libomp via Homebrew "
                "Note: Apple Silicon, use -arch x86_64 brew install X):"
            )
            print("`brew install libomp`")
            print("Then set the environment variables as follows:")
            print("For fish shell:")
            print(
                """
                        set -x CC gcc
                        set -x CXX g++
                        set -x CPPFLAGS "$CPPFLAGS -I/usr/local/include"
                        set -x CFLAGS "$CFLAGS -Wall"
                        set -x CXXFLAGS "$CXXFLAGS -Wall"
                        set -x LDFLAGS "$LDFLAGS -L/usr/local/lib"
                    """
            )
            print("For ZSH shell:")
            print(
                """
                        ENV CC=gcc
                        ENV CXX=g++
                        ENV CPPFLAGS="-I/usr/local/include"
                        ENV CFLAGS="-Wall"
                        ENV CXXFLAGS="-Wall"
                        ENV LDFLAGS="-L/usr/local/lib"
                    """
            )
            print("For Bash shell:")
            print(
                """
                        export CC=gcc
                        export CXX=g++
                        export CPPFLAGS="-I/usr/local/include"
                        export CFLAGS="-Wall"
                        export CXXFLAGS="-Wall"
                        export LDFLAGS="-L/usr/local/lib"
                    """
            )
        elif system_platform == "Linux":
            print("For Linux, you can install libomp via package manager, e.g.,")
            print("Ubuntu: `sudo apt install libomp-dev`")
            print("Fedora: `sudo dnf install libomp-devel`")
            print("Then set the environment variables as follows:")
            print("For fish shell:")
            print(
                """
                        set -x CC gcc
                        set -x CXX g++
                        set -x CPPFLAGS "$CPPFLAGS -I/usr/local/include"
                        set -x CFLAGS "$CFLAGS -Wall"
                        set -x CXXFLAGS "$CXXFLAGS -Wall"
                        set -x LDFLAGS "$LDFLAGS -L/usr/local/lib"
                    """
            )
            print("For ZSH shell:")
            print(
                """
                        ENV CC=gcc
                        ENV CXX=g++
                        ENV CPPFLAGS="-I/usr/local/include"
                        ENV CFLAGS="-Wall"
                        ENV CXXFLAGS="-Wall"
                        ENV LDFLAGS="-L/usr/local/lib"
                    """
            )
            print("For Bash shell:")
            print(
                """
                        export CC=gcc
                        export CXX=g++
                        export CPPFLAGS="-I/usr/local/include"
                        export CFLAGS="-Wall"
                        export CXXFLAGS="-Wall"
                        export LDFLAGS="-L/usr/local/lib"
                    """
            )

        return False

    print(f"All required compiler environment variables are set: {', '.join(found_variables)}")
    return True


def check_environment_variables() -> None:
    """
    Check for the presence of all required environment variables.

    This function checks for the necessary compiler and PDM environment variables
    based on the operating system. If any are missing, the program will exit.

    Note:
        For Windows, this function simply proceeds without checks.
    """
    system_platform: str = platform.system()
    all_variables_set: bool = True

    if system_platform in {"Darwin", "Linux"}:
        all_variables_set &= check_compiler_variables(system_platform)

    if all_variables_set:
        print("All required environment variables are set. Proceeding with the installation.")
    else:
        print("Some environment variables are missing. Please correct them before proceeding.")
        exit(1)

    if system_platform == "Windows":
        print("Running on Windows, Ignore environment variable checks.")
        print("Proceeding with the installation.")
    elif system_platform not in ["Darwin", "Linux"]:
        raise EnvironmentError(f"Unsupported operating system: {system_platform}. Exiting.")
