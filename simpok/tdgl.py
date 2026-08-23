import time

import numpy as np

from ._result import Result, make_meta


class TDGL:
    def __init__(self, n=256, dx=1.0, dt=0.1, h=0.0, eps=0.0,
                 dtype=np.float64):
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(
                f"dtype must be float32 or float64, got {dtype.name}"
            )
        if int(n) != n or n < 3:
            raise ValueError(f"n must be an integer >= 3, got {n!r}")
        if dx <= 0.0:
            raise ValueError(f"dx must be positive, got {dx!r}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps!r}")

        limit = 0.25 * dx * dx
        if dt > limit:
            raise ValueError(
                f"dt={dt} exceeds the explicit-Euler stability limit "
                f"dx^2/4={limit} for the 2-d 5-point Laplacian"
            )

        self.n = int(n)
        self.dx = float(dx)
        self.dt = float(dt)
        self.h = float(h)
        self.eps = float(eps)
        self.dtype = dtype

        self.psi = None
        self.seed = 0
        self.amplitude = 0.0
        self.step = 0
        self.backend = None

    def ic(self, seed=0, amplitude=0.01):
        if amplitude <= 0.0:
            raise ValueError(f"amplitude must be positive, got {amplitude!r}")

        rng = np.random.default_rng(seed)
        self.psi = rng.uniform(
            -amplitude, amplitude, size=(self.n, self.n)
        ).astype(self.dtype)
        self.seed = int(seed)
        self.amplitude = float(amplitude)
        self.step = 0
        return self

    @property
    def t(self):
        return self.step * self.dt

    def run(self, steps, nevery):
        if self.psi is None:
            raise RuntimeError("call ic() before run()")
        if np.shape(self.psi) != (self.n, self.n):
            raise ValueError(
                f"psi has shape {np.shape(self.psi)}, expected "
                f"{(self.n, self.n)}"
            )
        if int(steps) != steps or steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps!r}")
        if int(nevery) != nevery or nevery < 1:
            raise ValueError(
                f"nevery must be a positive integer, got {nevery!r}"
            )
        if nevery > steps:
            raise ValueError(f"nevery={nevery} exceeds steps={steps}")

        steps = int(steps)
        nevery = int(nevery)
        nsnap = steps // nevery
        step_start = self.step

        import mojo.importer  # noqa: F401

        from .mojo_module import _tdgl_run

        field = np.ascontiguousarray(self.psi, dtype=self.dtype)
        snaps = np.empty((nsnap, self.n, self.n), dtype=self.dtype)
        params = (
            self.n,
            steps,
            nevery,
            self.dt,
            self.dx,
            self.h,
            self.eps,
            self.seed,
            self.step,
            self.dtype == np.dtype(np.float32),
        )

        t0 = time.perf_counter()
        self.backend = _tdgl_run(field, snaps, params)
        elapsed = time.perf_counter() - t0

        self.psi = field
        self.step += steps

        meta = make_meta(
            "tdgl",
            n=self.n,
            dx=self.dx,
            dt=self.dt,
            h=self.h,
            eps=self.eps,
            dtype=self.dtype,
            seed=self.seed,
            amplitude=self.amplitude,
            steps=steps,
            nevery=nevery,
            step_start=step_start,
            step_end=self.step,
            backend=self.backend,
            elapsed=elapsed,
        )
        return Result(snaps, meta)
