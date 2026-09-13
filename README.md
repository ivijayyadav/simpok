# simpok

Phase-ordering kinetics in Python, with the computation in [Mojo](https://www.modular.com/mojo).

You write ordinary Python; the solvers run on your GPU if you have one, and on the CPU
otherwise, from the same code. The intent is to keep the modelling in a high-level API and push
the per-site arithmetic down to Mojo.

> **Status: alpha.** Two solvers, 2-d only, 5-point Laplacian, periodic boundaries. The API may
> still change.

## Install

```bash
pip install simpok
```

Python 3.10+. The Mojo toolchain arrives as a dependency — there is nothing else to install, and
no compiler to set up by hand.

**The first `import simpok` takes about 20 seconds.** It is compiling the Mojo sources for your
machine; the result is cached and every later import is instant. It recompiles only when the
package is upgraded. This is deliberate rather than a packaging shortcut: whether an accelerator
is targeted is decided at Mojo compile time, so building on your machine is what lets one
universal wheel use whatever hardware you actually have.

## Quick start

```python
from simpok import TDGL

sim = TDGL().ic(seed=0)
snaps, meta = sim.run(steps=90000, nevery=1000)

print(snaps.shape)          # (90, 256, 256)
print(meta["backend"])      # 'accelerator' or 'cpu'
print(meta["times"][-1])    # 9000.0
```

`run` returns a `Result` — a named tuple of `snaps` (a `(nsnap, n, n)` NumPy array) and `meta`
(a dict of every parameter used, plus `times`, `backend`, `device` and `elapsed`).

Conserved dynamics works the same way:

```python
from simpok import CHC

snaps, meta = CHC().ic(seed=0, psi0=-0.4).run(steps=50000, nevery=500)
```

Check what you're running on:

```python
from simpok import get_device
print(get_device())   # Device(available=True, name='NVIDIA RTX A5000', reason=None)
```

## The models

**`TDGL`** — nonconserved scalar order parameter (Model A). Domain coarsening in a quenched
ferromagnet; the Allen–Cahn growth law `L(t) ~ t^(1/2)`.

```
∂ψ/∂t = ψ − ψ³ + h + ∇²ψ + θ
```

**`CHC`** — conserved scalar order parameter (Model B). Spinodal decomposition in a binary
mixture; the Lifshitz–Slyozov growth law `L(t) ~ t^(1/3)`. The mean of ψ is conserved exactly,
so `psi0` sets the composition — `0.0` gives the bicontinuous critical quench, `-0.4` gives
minority droplets.

```
∂ψ/∂t = −∇²(ψ − ψ³ + ∇²ψ) + ∇·θ
```

Thermal noise is off by default (`eps=0`) — it is asymptotically irrelevant to
the growth laws. Set `eps > 0` to switch it on; for `CHC` it is applied as a bond-centred current
so that conservation stays exact to round-off.

## API

```python
TDGL(n=256, dx=1.0, dt=0.1,  h=0.0, eps=0.0, dtype=np.float64, device="auto")
CHC (n=256, dx=1.0, dt=0.01,        eps=0.0, dtype=np.float64, device="auto")
```

| Method | |
|---|---|
| `.ic(seed=0, amplitude=0.01)` | random initial condition; `CHC` also takes `psi0=0.0`. Returns `self`, so it chains. |
| `.run(steps, nevery)` | evolve `steps` steps, saving every `nevery`. Returns `Result(snaps, meta)`. |
| `.t` | current simulation time |
| `.psi` | current field, carried across calls — successive `run` calls continue the trajectory |

`device` is `"auto"` (default), `"accelerator"`, or `"cpu"`. `"accelerator"` raises if none is
usable rather than silently falling back. `dtype` is `float64` (default) or `float32`.

Timesteps are checked against the explicit-Euler stability limit at construction — `dt <= dx²/4`
for `TDGL`, and the much tighter fourth-order bound for `CHC`, which is why its default `dt` is
ten times smaller.

## Hardware notes

- **NVIDIA** — both dtypes work.
- **Apple Silicon** — Metal has no float64 at all, so use `dtype=np.float32` to run on the GPU.
  A float64 run falls back to the CPU automatically; asking for `device="accelerator"` with
  float64 raises with an explanation.
- **No GPU** — everything runs on the CPU with identical results. Deterministic (`eps=0`) runs
  are bit-identical between CPU and accelerator.

On float32, `CHC` is several times faster than float64 on consumer NVIDIA cards, whose
double-precision throughput is heavily reduced; the domain length scale agrees with float64 to
better than 0.01%. `TDGL` at small grids is limited by kernel-launch overhead rather than
arithmetic, so larger lattices use the GPU far more efficiently than small ones.

## Correctness

Both solvers are checked against analytic results, not just for plausibility:

- equilibrium interface `tanh(z/√2)`, second-order convergent in `dx`
- `TDGL`: droplet collapse `dR²/dt = −2(d−1)`; the `t^(1/2)` growth law
- `CHC`: the Cahn dispersion relation `σ(q) = q − q²` reproduced to 1 part in 10¹⁰; the mean
  conserved to 1 part in 10¹⁷; the `t^(1/3)` growth law
- conserved noise verified against the discrete fluctuation–dissipation relation
- seed reproducibility, resumability, and CPU/accelerator agreement

## Examples

`examples/` contains runnable scripts that produce snapshot figures:

```bash
python -m examples.tdgl_quickstart   # writes tdgl.png
python -m examples.chc_quickstart    # writes chc.png
```

They need matplotlib: `pip install simpok[examples]`.

## Reference

Sanjay Puri, "Kinetics of Phase Transitions", Ch. 1 in *Kinetics of Phase Transitions*,
S. Puri and V. Wadhawan (eds.), CRC Press (2009).

## License

MIT — see [LICENSE](LICENSE).
