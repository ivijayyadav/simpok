import warnings
from typing import NamedTuple

import numpy as np

_DEVICE_NAME = None
_DEVICE_CODES = {"gpu": 1, "cpu": 2}


class Result(NamedTuple):
    snaps: np.ndarray
    meta: dict


def device_name():
    global _DEVICE_NAME

    if _DEVICE_NAME is None:
        from .device import get_device

        _DEVICE_NAME = get_device().name or "cpu"
    return _DEVICE_NAME


def call_solver(fn, arrays, params):
    try:
        return fn(*arrays, params)
    except Exception as e:
        if params[-1] != _DEVICE_CODES["gpu"]:
            raise
        warnings.warn(
            f"{e}; falling back to the CPU", RuntimeWarning, stacklevel=3
        )
        return fn(*arrays, params[:-1] + (_DEVICE_CODES["cpu"],))


def device_code(device):
    if device is None:
        return 0
    try:
        return _DEVICE_CODES[device]
    except (KeyError, TypeError):
        raise ValueError(
            f'device must be None, "gpu" or "cpu", got {device!r}'
        ) from None


def make_meta(solver, *, grid, dx, dt, dtype, seed, amplitude, steps,
              nevery, step_start, step_end, backend, device, elapsed,
              **physics):
    nsnap = steps // nevery
    meta = {"solver": solver, **grid, "dx": dx, "dt": dt}
    meta.update(physics)
    meta.update(
        {
            "dtype": np.dtype(dtype).name,
            "seed": seed,
            "amplitude": amplitude,
            "steps": steps,
            "nevery": nevery,
            "nsnap": nsnap,
            "step_start": step_start,
            "step_end": step_end,
            "times": (step_start + np.arange(1, nsnap + 1) * nevery) * dt,
            "backend": backend,
            "device": device_name(),
            "device_request": device,
            "elapsed": elapsed,
        }
    )
    return meta
