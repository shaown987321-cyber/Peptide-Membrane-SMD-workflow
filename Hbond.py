import argparse
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("xvg", nargs="?", default="hbnum.xvg")
p.add_argument("-o", "--output", default="hbond")
p.add_argument("--title", default="Hydrogen Bond Analysis")
p.add_argument("--window", type=float, default=5.0, help="Reporting window in ns")
p.add_argument("--start", type=float, default=None, help="Mean/SD start time in ns")
p.add_argument("--dpi", type=int, default=600)
a = p.parse_args()

path = a.xvg
if not os.path.isfile(path):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.xvg)
if not os.path.isfile(path):
    sys.exit(f"Input file not found: {a.xvg}")
out = a.output if os.path.dirname(a.output) else os.path.join(os.path.dirname(os.path.abspath(path)), a.output)

rows = []
with open(path) as f:
    for line in f:
        if line.strip() and line.lstrip()[0] not in "#@":
            try:
                rows.append([float(x) for x in line.split()[:2]])
            except ValueError:
                continue
if len(rows) < 2:
    sys.exit("XVG must contain at least two valid data rows.")

t, n = np.asarray(rows).T
if np.any(np.diff(t) <= 0) or a.window <= 0:
    sys.exit("Time must increase strictly and window must be positive.")

print(f"Input: {path}")
print(f"Overall mean H-bonds: {n.mean():.3f}")
print(f"Overall SD: {n.std(ddof=1):.3f}")

if a.start is not None:
    sel = n[t >= a.start]
    if len(sel) < 2:
        sys.exit("Need at least two data points after --start.")
    print(f"Mean from {a.start:g} to {t[-1]:g} ns: {sel.mean():.3f}")
    print(f"SD from {a.start:g} to {t[-1]:g} ns: {sel.std(ddof=1):.3f}")

dt = np.median(np.diff(t))
mask = (t >= t[0]) & (t <= t[0] + a.window)
if mask.sum() >= 2:
    print(f"First {a.window:g} ns mean: {n[mask].mean():.3f}")
    print(f"First {a.window:g} ns SD: {n[mask].std(ddof=1):.3f}")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.linewidth": 1.1,
    "xtick.direction": "in", "ytick.direction": "in",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(7.2, 4))
ax.plot(t, n, color="#9ecae1", lw=0.7)
ax.set(xlim=(t[0], t[-1]), ylim=(0, None),
       xlabel="Time (ns)", ylabel="Number of hydrogen bonds")
ax.set_title(a.title, fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="y", color="0.88", lw=0.6)
ax.set_axisbelow(True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(out + ".png", dpi=a.dpi, bbox_inches="tight", facecolor="white")
fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out}.png and {out}.pdf")