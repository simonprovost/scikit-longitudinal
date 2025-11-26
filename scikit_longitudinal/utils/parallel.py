from importlib import import_module, util
from types import ModuleType
from typing import Optional

_PARALLEL_EXTRA = "Scikit-longitudinal[parallelisation]"


def _missing_ray_error() -> ImportError:
    return ImportError(
        "Ray is required for parallel execution. "
        f"Install the 'parallelisation' extra: pip install {_PARALLEL_EXTRA}."
    )


def get_ray_or_raise() -> ModuleType:
    """Import and return `ray`, raising a clear message when absent."""

    if util.find_spec("ray") is None:
        raise _missing_ray_error()
    return import_module("ray")


def get_ray_for_parallel(parallel: bool, num_cpus: Optional[int] = None) -> Optional[ModuleType]:
    """Return an initialised Ray module when parallel processing is requested.

    Parameters:
        parallel:
            Whether the caller intends to use Ray-backed parallelism.
        num_cpus:
            Number of CPUs to request when initialising Ray. ``None`` or ``-1``
            delegates to Ray's defaults.

    Returns:
        ModuleType or None
            The Ray module when ``parallel`` is True; otherwise ``None``.

    Raises:
        ImportError
            If Ray is not installed when ``parallel`` is True.
    """

    if not parallel:
        return None

    ray = get_ray_or_raise()
    if not ray.is_initialized():
        if num_cpus is not None and num_cpus != -1:
            ray.init(num_cpus=num_cpus)
        else:
            ray.init()
    return ray
