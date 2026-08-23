from typing import NamedTuple

import numpy as np

_DEVICE_NAME = None
_DEVICE_CODES = {"auto": 0, "accelerator": 1, "gpu": 1, "cpu": 2}


class Result(NamedTuple):
    snaps: np.ndarray
    meta: dict


def device_name():
    global _DEVICE_NAME

    if _DEVICE_NAME is None:
        from .device import get_device

        _DEVICE_NAME = get_device().name or "cpu"
    return _DEVICE_NAME


def device_code(device):
    try:
        return _DEVICE_CODES[device]
    except (KeyError, TypeError):
        raise ValueError(
            f"device must be one of {sorted(_DEVICE_CODES)}, got {device!r}"
        ) from None


def make_meta(solver, *, n, dx, dt, dtype, seed, amplitude, steps, nevery,
              step_start, step_end, backend, device, elapsed, **physics):
    nsnap = steps // nevery
    meta = {"solver": solver, "n": n, "dx": dx, "dt": dt}
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
