
#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("input", nargs="?", default="TAT_RMSF.xvg")
parser.add_argument("-o", "--output", default="TAT_RMSF")
args = parser.parse_args()

# TAT sequence: GRKKRRQRRRPQ
residue_names = [
    "GLY1", "ARG2", "LYS3", "LYS4",
    "ARG5", "ARG6", "GLN7", "ARG8",
    "ARG9", "ARG10", "PRO11", "GLN12"
]

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
    raise ValueError("No valid RMSF data found in the XVG file.")

data = np.asarray(data)
residue, rmsf = data[:, 0], data[:, 1]

if len(data) != len(residue_names):
    raise ValueError(
        f"Expected 12 TAT residues, but found {len(data)} RMSF values. "
        "Check residue selection and numbering."
    )

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

fig, ax = plt.subplots(figsize=(7.5, 4.0))
ax.plot(
    np.arange(12), rmsf,
    color="#245A81", linewidth=1.2,
    marker="o", markersize=3.5
)

ax.set_xlabel("TAT Residue")
ax.set_ylabel("RMSF (nm)")
ax.set_xticks(np.arange(12))
ax.set_xticklabels(residue_names, rotation=45, ha="right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(width=0.8, length=3)

fig.tight_layout()
out = Path(args.input).resolve().parent / args.output
fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
plt.show()

print(f"Mean RMSF: {np.mean(rmsf):.4f} nm")
print(f"Maximum RMSF: {np.max(rmsf):.4f} nm")
print(f"Maximum fluctuation: {residue_names[np.argmax(rmsf)]}")
print(f"Saved: {out}.png and {out}.pdf")
