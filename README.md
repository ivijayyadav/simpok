# simpok

High level python API for Phase field simulation of phase-ordering kinetics.

Implemented: Dimensionless CHC, TDGL, and SDSD solvers, finite differences in space and explicit Euler in time.

## Install

```bash
pip install simpok
```

Python 3.10+. The Mojo toolchain arrives as a dependency — there is nothing else to install, and
no compiler to set up by hand.

**The first `import simpok` takes about 20 seconds.** It is compiling the Mojo sources for your
machine; the result is cached and every later import is instant. It recompiles only when the
package is upgraded.

## Quick start

```python
from simpok import TDGL

sim = TDGL().ic(seed=0)
snaps, meta = sim.run(steps=90000, nevery=1000)

print(snaps.shape)          # (90, 256, 256)
print(meta["backend"])      # 'gpu' or 'cpu'
print(meta["times"][-1])    # 9000.0
```

`run` returns a `Result` — a named tuple of `snaps` (a `(nsnap, ny, nx)` NumPy array) and `meta`
(a dict of every parameter used, plus `times`, `backend`, `device` and `elapsed`).

Conserved dynamics works the same way:

```python
from simpok import CHC

snaps, meta = CHC().ic(seed=0, psi0=-0.4).run(steps=50000, nevery=500)
```

And with a wall that prefers one component:

```python
from simpok import SDSD

sim = SDSD(nz=256, nx=512, eps=0.041)
snaps, meta = sim.ic(seed=0, psi0=-0.4).run(steps=80000, nevery=2000)

profile = snaps.mean(axis=2)   # laterally averaged depth profile, (nsnap, nz)
```

Check what you're running on:

```python
from simpok import get_device

d = get_device()
print(d)        # Device(available=True, name='NVIDIA RTX A5000', reason=None, kind='gpu')
print(d.kind)   # 'gpu' or 'cpu' — pass it straight back as device=
```

## The models

**`TDGL`** — nonconserved scalar order parameter (Model A).

```
∂ψ/∂t = ψ − ψ³ + h + ∇²ψ + θ
```

**`CHC`** — conserved scalar order parameter (Model B).

```
∂ψ/∂t = −∇²(ψ − ψ³ + ∇²ψ) + ∇·θ
```

**`SDSD`** — surface-directed spinodal decomposition. The same conserved dynamics in contact with
a wall at `z = 0` that prefers one component: the bulk phase separates while the wall is wetted,
and the two processes interfere. The result is a layered composition profile — wetting layer,
depletion layer, then bulk — that propagates away from the wall.

```
∂ψ/∂t = −∇²(ψ − ψ³ + ½∇²ψ − V(z)) + ∇·θ                    z > 0

h₁ + gψ + γ ∂ψ/∂z = 0                                      at z = 0
∂/∂z(−ψ + ψ³ − ½∇²ψ + V) = 0                               at z = 0
```

with the long-ranged wall potential `V(z) = −V₀/zⁿ` (cut off at `−V₀` for `z ≤ 1`).

## API

```python
TDGL(ny=256, nx=256, dx=1.0, dt=0.1, h=0.0, eps=0.0, dtype=np.float64, device=None)
CHC (ny=256, nx=256, dx=1.0, dt=0.01, eps=0.0, dtype=np.float64, device=None)
SDSD(nz=256, nx=256, dx=1.0, dt=0.03, h1=0.8, g=-0.4, gamma=0.4, v0=None, n=4,
     potential=None, eps=0.0, dtype=np.float64, device=None)
```

The wall potential does not have to be that power law. `potential=` takes any function of depth
(or a ready-made array), evaluated once on `sim.z` and stored as `sim.V`:

```python
SDSD(potential=lambda z: -0.8 * np.exp(-z / 3.0))        # short-ranged well
SDSD(potential=lambda z: np.where(z <= 5.0, -0.8, 0.0))  # square well
SDSD(potential=np.zeros(256))                            # no potential at all
```

## Hardware notes

- **Supported Device** — GPU (AMD, NVIDIA, APPLE) and CPU

## License

MIT — see [LICENSE](LICENSE).
