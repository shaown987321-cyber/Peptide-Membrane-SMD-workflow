
#!/usr/bin/env python3
"""Publication-quality GROMACS RMSD plot: PNG, PDF, CSV."""
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("input", nargs="?", default="TAT_RMSD.xvg")
p.add_argument("-o", "--output", help="Output filename base")
p.add_argument("--time-unit", choices=["fs", "ps", "ns", "us"])
p.add_argument("--angstrom", action="store_true", help="Convert nm to Angstrom")
p.add_argument("--title", help="Optional plot title")
p.add_argument("--x-interval", type=float, default=100, help="X tick interval in ns")
args = p.parse_args()

path = Path(args.input)
if not path.is_file():
    sys.exit(f"ERROR: File not found: {path}")
try:
    header = path.read_text(encoding="utf-8", errors="replace")
    data = np.loadtxt(path, comments=("#", "@", "&"),
                      usecols=(0, 1), ndmin=2)
except (OSError, ValueError) as exc:
    sys.exit(f"ERROR: Cannot read XVG data: {exc}")

data = data[np.isfinite(data).all(axis=1)]
if len(data) == 0:
    sys.exit("ERROR: No valid numerical data found.")

m = re.search(r'xaxis\s+label\s+"[^"]*\(\s*(fs|ps|ns|us)\s*\)"',
              header, re.I)
unit = args.time_unit or (m.group(1).lower() if m else "ps")
if not args.time_unit and not m:
    print("WARNING: Time unit missing; assuming ps.")
if args.x_interval <= 0:
    sys.exit("ERROR: X-axis interval must be positive.")

time_ns = data[:, 0] * {"fs": 1e-6, "ps": 1e-3,
                         "ns": 1.0, "us": 1e3}[unit]
rmsd = data[:, 1] * (10 if args.angstrom else 1)
yu = "Å" if args.angstrom else "nm"
order = np.argsort(time_ns)
time_ns, rmsd = time_ns[order], rmsd[order]
tmax = float(time_ns.max())

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42
})

fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(time_ns, rmsd, color="#0072B2", linewidth=0.85,
        rasterized=True)
ax.set_xlabel("Time (ns)", labelpad=5)
ax.set_ylabel(f"Backbone RMSD ({yu})", labelpad=5)
if args.title:
    ax.set_title(args.title, fontsize=10, pad=8)

ax.set_xlim(0, tmax if tmax > 0 else args.x_interval)
ymin = min(0.0, float(rmsd.min()))
ymax = float(rmsd.max())
upper = ymax * 1.10 if ymax > 0 else 1.0
ax.set_ylim(ymin, upper)
ax.xaxis.set_major_locator(MultipleLocator(args.x_interval))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="out", length=3, width=0.7, pad=3)
fig.tight_layout()

base = Path(args.output) if args.output else path.with_suffix("")
base = base.parent / (base.name + "_plot")
try:
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    np.savetxt(str(base) + ".csv", np.c_[time_ns, rmsd],
               delimiter=",", fmt="%.6f",
               header=f"time_ns,rmsd_{'angstrom' if args.angstrom else 'nm'}",
               comments="")
except OSError as exc:
    sys.exit(f"ERROR: Could not save output: {exc}")
finally:
    plt.close(fig)

print("\n========== RMSD SUMMARY ==========")
print(f"Input: {path}")
print(f"Frames: {len(rmsd)}")
print(f"Time range: {time_ns.min():.2f}–{tmax:.2f} ns")
print(f"Y-axis range: {ymin:.3f}–{upper:.3f} {yu}")
print(f"Mean RMSD: {np.mean(rmsd):.4f} {yu}")
print(f"Median RMSD: {np.median(rmsd):.4f} {yu}")
print(f"Minimum RMSD: {rmsd.min():.4f} {yu}")
print(f"Maximum RMSD: {rmsd.max():.4f} {yu}")
print(f"Saved: {base}.png, {base}.pdf, {base}.csv")
