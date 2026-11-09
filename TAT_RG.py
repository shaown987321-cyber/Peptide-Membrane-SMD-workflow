
#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot GROMACS radius of gyration.")
parser.add_argument("input", nargs="?", default="TAT_Rg.xvg")
parser.add_argument("-o", "--output", default="TAT_Rg")
parser.add_argument("--window", type=float, default=100,
                    help="Final time window in ns (default: 100)")
args = parser.parse_args()

data = []
with open(args.input, encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line and not line.startswith(("#", "@")):
            try:
                values = line.split()
                data.append([float(values[0]), float(values[1])])
            except (ValueError, IndexError):
                continue

if not data:
    raise ValueError("No valid data found in the XVG file.")

data = np.asarray(data)
time_ns = data[:, 0] / 1000.0
rg = data[:, 1]

if args.window <= 0:
    raise ValueError("--window must be greater than zero.")

mask = time_ns >= time_ns.max() - args.window
if not np.any(mask):
    raise ValueError("No data found in the requested final time window.")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(time_ns, rg, color="#245A81", linewidth=0.8)

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Radius of gyration (nm)")
ax.set_xlim(time_ns.min(), time_ns.max())
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(width=0.8, length=3)
fig.tight_layout()

out = Path(args.input).resolve().parent / args.output
fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
plt.show()

print(f"Total simulation time: {time_ns.max():.2f} ns")
print(f"Mean Rg (whole trajectory): {np.mean(rg):.4f} nm")
print(f"Mean Rg (final {args.window:g} ns): {np.mean(rg[mask]):.4f} nm")
print(f"SD Rg (whole trajectory): {np.std(rg):.4f} nm")
print(f"Saved: {out}.png and {out}.pdf")
