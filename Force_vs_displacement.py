
import numpy as np
import matplotlib.pyplot as plt

# Input files
PULLX_FILE = "SMD_pullx.xvg"
PULLF_FILE = "SMD_pullf.xvg"


def load_xvg(filename):
    data = np.loadtxt(filename, comments=("#", "@"))
    return data[:, 0], data[:, 1]


# Load coordinate and force
time_x, position = load_xvg(PULLX_FILE)
time_f, force = load_xvg(PULLF_FILE)

# Verify matching time points
if len(time_x) != len(time_f) or not np.allclose(time_x, time_f):
    raise ValueError("Time points do not match between XVG files.")

# Use actual pulling coordinate
x_coordinate = position

# Print statistics
print("\nSMD Mechanical Analysis")
print("-----------------------")
print(f"Number of frames: {len(force)}")
print(f"Initial coordinate: {position[0]:.5f} nm")
print(f"Final coordinate: {position[-1]:.5f} nm")
print(f"Total coordinate change: {position[-1] - position[0]:.5f} nm")
print(f"Mean signed force: {np.mean(force):.3f} kJ mol-1 nm-1")
print(f"Force standard deviation: {np.std(force):.3f} kJ mol-1 nm-1")
print(f"Maximum force: {np.max(force):.3f} kJ mol-1 nm-1")
print(f"Minimum force: {np.min(force):.3f} kJ mol-1 nm-1")

# Publication-style settings
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 600,
    "pdf.fonttype": 42
})

# Create figure
fig, ax = plt.subplots(figsize=(3.5, 2.8))

ax.plot(x_coordinate, force, color="#4682B4", linewidth=0.5)

ax.set_xlabel("Pulling coordinate (nm)")
ax.set_ylabel(r"Pulling force (kJ mol$^{-1}$ nm$^{-1}$)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(width=0.8, length=3)

fig.tight_layout()

# Save figures
fig.savefig("Force_vs_Actual_Coordinate.png", dpi=600, bbox_inches="tight")
fig.savefig("Force_vs_Actual_Coordinate.pdf", bbox_inches="tight")

print("\nFigures saved successfully.")

plt.show()
