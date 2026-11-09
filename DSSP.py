
#!/usr/bin/env python3
"""Publication-quality DSSP analysis for the 12-residue TAT peptide."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent
DSSP_FILE = BASE / "TAT_ss.dat"
XVG_FILE = BASE / "TAT_ss_num.xvg"
OUT = BASE / "DSSP_results"
OUT.mkdir(parents=True, exist_ok=True)

RESIDUES = [
    "GLY1", "ARG2", "LYS3", "LYS4", "ARG5", "ARG6",
    "GLN7", "ARG8", "ARG9", "ARG10", "PRO11", "GLN12"
]

STRUCTURES = {
    "H": "α-Helix", "G": "3₁₀-Helix", "I": "π-Helix",
    "E": "β-Strand", "B": "β-Bridge", "P": "PPII-Helix",
    "T": "Turn", "S": "Bend", "~": "Loop", "=": "Break"
}
CODES = list(STRUCTURES)
COLORS = [
    "#D73027", "#FC8D59", "#FEE08B", "#4575B4", "#74ADD1",
    "#984EA3", "#66BD63", "#A6D96A", "#D9D9D9", "#666666"
]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(len(CODES) + 1) - 0.5, len(CODES))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.linewidth": 0.8, "xtick.direction": "out",
    "ytick.direction": "out", "pdf.fonttype": 42,
    "savefig.dpi": 600
})

# Read and validate DSSP frames.
if not DSSP_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {DSSP_FILE}")

frames = [
    line.strip() for line in DSSP_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith(("#", "@"))
]
if not frames:
    raise ValueError("TAT_ss.dat contains no DSSP frames.")
if any(len(row) != len(RESIDUES) for row in frames):
    raise ValueError("Each DSSP frame must contain exactly 12 residue codes.")

unknown = sorted(set("".join(frames)) - set(CODES))
if unknown:
    raise ValueError(f"Unknown DSSP codes found: {unknown}")

ss = np.array([list(row) for row in frames])
numeric = np.array([[CODES.index(c) for c in row] for row in ss])
nframes, nres = ss.shape

# Read simulation times from the XVG file.
times = []
if XVG_FILE.exists():
    for line in XVG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        try:
            times.append(float(parts[0]))
        except (ValueError, IndexError):
            continue

if len(times) != nframes:
    print("Warning: XVG time count does not match DSSP frames.")
    print("Using an evenly spaced 0–500 ns time axis.")
    times = np.linspace(0, 500, nframes)
else:
    times = np.asarray(times)

# Save per-frame residue assignments.
frame_df = pd.DataFrame(ss, columns=RESIDUES)
frame_df.insert(0, "Time_ns", times)
frame_df.to_csv(OUT / "TAT_DSSP_per_frame.csv", index=False)

# Calculate residue-specific secondary-structure occupancy.
occupancy = np.array([
    [100 * np.mean(ss[:, r] == code) for code in CODES]
    for r in range(nres)
])
occ_df = pd.DataFrame(occupancy, index=RESIDUES,
                      columns=[STRUCTURES[c] for c in CODES])
occ_df.index.name = "Residue"
occ_df.to_csv(OUT / "TAT_DSSP_occupancy.csv")

overall = pd.DataFrame({
    "Secondary_structure": [STRUCTURES[c] for c in CODES],
    "Percentage": [100 * np.mean(ss == c) for c in CODES]
})
overall.to_csv(OUT / "TAT_DSSP_overall_percentages.csv", index=False)

# Legend labels show scientific names only, never DSSP code letters.
legend_handles = [
    Patch(facecolor=COLORS[i], edgecolor="black", linewidth=0.4,
          label=STRUCTURES[code])
    for i, code in enumerate(CODES)
]

# 1. Residue-by-time categorical heatmap.
fig, ax = plt.subplots(figsize=(11, 5.8))
im = ax.imshow(
    numeric.T, aspect="auto", interpolation="nearest",
    cmap=CMAP, norm=NORM,
    extent=[times[0], times[-1], nres - 0.5, -0.5]
)
ax.set_yticks(range(nres), RESIDUES)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Residue")
ax.set_title("TAT Peptide Secondary Structure")
# Reserve space on the right so the legend cannot cover the data.
fig.subplots_adjust(right=0.76, left=0.12, bottom=0.13, top=0.90)
ax.legend(
    handles=legend_handles, title="Secondary Structure",
    loc="upper left", bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0, frameon=True, framealpha=1,
    edgecolor="black", fontsize=8, title_fontsize=8,
    labelspacing=0.55
)
fig.savefig(OUT / "TAT_DSSP_heatmap.png", bbox_inches="tight")
fig.savefig(OUT / "TAT_DSSP_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# 2. Residue-by-structure occupancy heatmap.
fig, ax = plt.subplots(figsize=(12, 5.5))
im = ax.imshow(occupancy, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
ax.set_yticks(range(nres), RESIDUES)
ax.set_xticks(range(len(CODES)), [STRUCTURES[c] for c in CODES],
              rotation=35, ha="right")
ax.set_xlabel("Secondary structure")
ax.set_ylabel("Residue")
ax.set_title("Secondary-Structure Occupancy by Residue")
for r in range(nres):
    for c in range(len(CODES)):
        color = "white" if occupancy[r, c] >= 55 else "black"
        ax.text(c, r, f"{occupancy[r, c]:.1f}",
                ha="center", va="center", fontsize=7, color=color)
fig.colorbar(im, ax=ax, label="Occupancy (%)", pad=0.03)
fig.tight_layout()
fig.savefig(OUT / "TAT_DSSP_occupancy_heatmap.png", bbox_inches="tight")
fig.savefig(OUT / "TAT_DSSP_occupancy_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# 3. Overall secondary-structure percentage bar chart.
plot_df = overall.sort_values("Percentage", ascending=True)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(plot_df["Secondary_structure"], plot_df["Percentage"],
        color="#4575B4", edgecolor="black", linewidth=0.4)
ax.set_xlabel("Fraction of all residue-frame assignments (%)")
ax.set_ylabel("Secondary structure")
ax.set_title("Overall TAT Secondary-Structure Distribution")
ax.set_xlim(0, max(5, plot_df["Percentage"].max() * 1.15))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "TAT_DSSP_percentages.png", bbox_inches="tight")
fig.savefig(OUT / "TAT_DSSP_percentages.pdf", bbox_inches="tight")
plt.close(fig)

# 4. Secondary-structure time series.
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, code in enumerate(CODES):
    values = 100 * np.mean(ss == code, axis=1)
    ax.plot(times, values, color=COLORS[i], linewidth=1.0,
            label=STRUCTURES[code])
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Residue-frame assignments (%)")
ax.set_title("Secondary-Structure Population vs Time")
ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Put legend outside the upper-right of the axes, with a clear gap.
fig.subplots_adjust(right=0.76, left=0.10, bottom=0.13, top=0.90)
ax.legend(
    title="Secondary Structure",
    loc="upper left", bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0, frameon=True, framealpha=1,
    edgecolor="black", fontsize=8, title_fontsize=8,
    labelspacing=0.55
)
fig.savefig(OUT / "TAT_DSSP_timeseries.png", bbox_inches="tight")
fig.savefig(OUT / "TAT_DSSP_timeseries.pdf", bbox_inches="tight")
plt.close(fig)

print("\nDSSP analysis completed successfully.")
print(f"Frames analyzed: {nframes}")
print(f"Simulation time: {times[0]:.3f}–{times[-1]:.3f} ns")
print(f"Results folder: {OUT}")
print("\nOverall secondary-structure percentages:")
print(overall.to_string(index=False, formatters={"Percentage": "{:.2f}".format}))
print("\nResidue-specific occupancy CSV:")
print(OUT / "TAT_DSSP_occupancy.csv")
