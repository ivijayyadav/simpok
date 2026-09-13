import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simpok import CHC
import numpy as np

steps = 500000
nevery = 5000
device = "auto"

sim = CHC(device=device).ic(seed=0)
snaps, meta = sim.run(steps=steps, nevery=nevery)

t = meta["times"]

for key in ("solver", "backend", "device", "dtype", "n", "dx", "dt",
            "eps", "psi0", "seed", "steps", "nevery", "device_request"):
    print(f"{key:<14}: {meta[key]}")

print(f"{'snapshots':<14}: {snaps.shape}")
print(f"{'elapsed':<14}: {meta['elapsed']:.3f} s (includes first-call warmup)")

fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
n_snaps = snaps.shape[0]
for ax, k in zip(axes, (4, 19, 49, n_snaps-1)):
    im = ax.imshow(
        snaps[k], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
    )
    ax.set_title(f"$t = {t[k]:.0f}$", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(
    im, ax=axes, orientation="vertical", fraction=0.02, pad=0.012,
    ticks=[-1, 0, 1],
)
cbar.set_label(r"$\psi$", fontsize=11, rotation=0, labelpad=10)
cbar.ax.tick_params(labelsize=9)

out = os.path.abspath("chc.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nwrote {out}")
