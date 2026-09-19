import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simpok import SDSD
import numpy as np

steps = 80000
nevery = 2000
psi0 = 0.0
eps = 0.041

sim = SDSD(nz=300, nx=400, eps=eps)
snaps, meta = sim.ic(seed=0, amplitude=0.1, psi0=psi0).run(
    steps=steps, nevery=nevery
)

t = meta["times"]

for key in ("solver", "backend", "device", "dtype", "nz", "nx", "dx", "dt",
            "h1", "g", "gamma", "v0", "n", "eps", "psi0", "seed", "steps",
            "nevery", "device_request"):
    print(f"{key:<14}: {meta[key]}")

print(f"{'snapshots':<14}: {snaps.shape}")
print(f"{'elapsed':<14}: {meta['elapsed']:.3f} s (includes first-call warmup)")

panels = (0, 3, 19, len(t) - 1)
panel_w = 3.4
aspect = sim.nz / sim.nx
fig, axes = plt.subplots(
    2, 4, figsize=(4 * panel_w, panel_w * (aspect + 0.62)),
    gridspec_kw={"height_ratios": [aspect, 0.52]}, layout="constrained",
)

for col, k in enumerate(panels):
    im = axes[0, col].imshow(
        snaps[k], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
    )
    axes[0, col].set_title(f"$t = {t[k]:.0f}$", fontsize=11)
    axes[0, col].set_xticks([])
    axes[0, col].set_yticks([])

    axes[1, col].plot(sim.z, snaps[k].mean(axis=1), lw=1.1, color="C3")
    axes[1, col].axhline(psi0, color="k", lw=0.6)
    axes[1, col].set_xlim(0, 80)
    axes[1, col].set_ylim(-1.2, 1.4)
    axes[1, col].set_xlabel("$z$", fontsize=10)
    if col == 0:
        axes[1, col].set_ylabel(r"$\psi_{\rm av}(z,t)$", fontsize=10)
    else:
        axes[1, col].set_yticklabels([])

cbar = fig.colorbar(
    im, ax=axes[0], orientation="vertical", fraction=0.02, pad=0.012,
    ticks=[-1, 0, 1],
)
cbar.set_label(r"$\psi$", fontsize=11, rotation=0, labelpad=10)
cbar.ax.tick_params(labelsize=9)

out = os.path.abspath("sdsd.png")
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
