import time
import numpy as np
from ._result import Result, call_solver, device_code, make_meta

class SDSD:
    def __init__(self, nz=256, nx=256, dx=1.0, dt=0.03, h1=0.8, g=-0.4,
                 gamma=0.4, v0=None, n=4, potential=None, eps=0.0,
                 dtype=np.float64, device=None):
        dtype = np.dtype(dtype)
        self._device_code = device_code(device)
        self.device = device
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(
                f"dtype must be float32 or float64, got {dtype.name}"
            )
        if int(nz) != nz or nz < 5:
            raise ValueError(f"nz must be an integer >= 5, got {nz!r}")
        if int(nx) != nx or nx < 5:
            raise ValueError(f"nx must be an integer >= 5, got {nx!r}")
        if dx <= 0.0:
            raise ValueError(f"dx must be positive, got {dx!r}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps!r}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma!r}")
        if n <= 0.0:
            raise ValueError(f"n must be positive, got {n!r}")
        if gamma / dx - g == 0.0:
            raise ValueError(
                f"gamma/dx - g must be non-zero, got gamma={gamma}, "
                f"dx={dx}, g={g}; the surface boundary condition is "
                f"degenerate there"
            )

        q = 8.0 / (dx * dx)
        growth = 0.5 * q * q - q
        if growth > 0.0:
            limit = 2.0 / growth
            if dt > limit:
                raise ValueError(
                    f"dt={dt} exceeds the explicit-Euler stability limit "
                    f"{limit} for the 2-d 5-point biharmonic at dx={dx}"
                )

        self.nz = int(nz)
        self.nx = int(nx)
        self.dx = float(dx)
        self.dt = float(dt)
        self.h1 = float(h1)
        self.g = float(g)
        self.gamma = float(gamma)
        self.v0 = float(h1) if v0 is None else float(v0)
        self.n = float(n)
        self.eps = float(eps)
        self.dtype = dtype
        self.potential_kind = "power-law" if potential is None else "custom"
        self.V = self.build_potential(potential)

        self.psi = None
        self.seed = 0
        self.amplitude = 0.0
        self.psi0 = 0.0
        self.step = 0
        self.backend = None

    @property
    def shape(self):
        return (self.nz, self.nx)

    @property
    def z(self):
        return (self.nz - np.arange(self.nz)) * self.dx

    def power_law(self, z):
        return -np.where(z <= 1.0, self.v0, self.v0 / z**self.n)

    def build_potential(self, potential=None):
        z = self.z
        if potential is None:
            v = self.power_law(z)
        elif callable(potential):
            v = np.asarray(potential(z), dtype=np.float64)
            if v.ndim == 0:
                v = np.full(self.nz, float(v))
        else:
            v = np.asarray(potential, dtype=np.float64)
        if np.shape(v) != (self.nz,):
            raise ValueError(
                f"potential must give one value per row, expected shape "
                f"{(self.nz,)}, got {np.shape(v)}"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError("potential must be finite at every z")
        return np.ascontiguousarray(v, dtype=self.dtype)

    def ic(self, seed=0, amplitude=0.01, psi0=0.0):
        if amplitude <= 0.0:
            raise ValueError(f"amplitude must be positive, got {amplitude!r}")

        rng = np.random.default_rng(seed)
        self.psi = (
            psi0 + rng.uniform(-amplitude, amplitude, size=(self.nz, self.nx))
        ).astype(self.dtype)
        self.seed = int(seed)
        self.amplitude = float(amplitude)
        self.psi0 = float(psi0)
        self.step = 0
        return self

    @property
    def t(self):
        return self.step * self.dt

    def run(self, steps, nevery):
        if self.psi is None:
            raise RuntimeError("call ic() before run()")
        if np.shape(self.psi) != (self.nz, self.nx):
            raise ValueError(
                f"psi has shape {np.shape(self.psi)}, expected "
                f"{(self.nz, self.nx)}"
            )
        if np.shape(self.V) != (self.nz,):
            raise ValueError(
                f"V has shape {np.shape(self.V)}, expected {(self.nz,)}"
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

        from .mojo_module import _sdsd_run

        field = np.ascontiguousarray(self.psi, dtype=self.dtype)
        pot = np.ascontiguousarray(self.V, dtype=self.dtype)
        snaps = np.empty((nsnap, self.nz, self.nx), dtype=self.dtype)
        params = (
            self.nz,
            self.nx,
            steps,
            nevery,
            self.dt,
            self.dx,
            self.h1,
            self.g,
            self.gamma,
            self.eps,
            self.seed,
            self.step,
            self.dtype == np.dtype(np.float32),
            self._device_code,
        )

        t0 = time.perf_counter()
        self.backend = call_solver(_sdsd_run, (field, pot, snaps), params)
        elapsed = time.perf_counter() - t0

        self.psi = field
        self.step += steps

        meta = make_meta(
            "sdsd",
            grid={"nz": self.nz, "nx": self.nx},
            dx=self.dx,
            dt=self.dt,
            h1=self.h1,
            g=self.g,
            gamma=self.gamma,
            v0=self.v0,
            n=self.n,
            potential=self.potential_kind,
            eps=self.eps,
            psi0=self.psi0,
            dtype=self.dtype,
            seed=self.seed,
            amplitude=self.amplitude,
            steps=steps,
            nevery=nevery,
            step_start=step_start,
            step_end=self.step,
            backend=self.backend,
            device=self.device,
            elapsed=elapsed,
        )
        return Result(snaps, meta)
