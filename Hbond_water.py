
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# FILE SETTINGS
# =====================================================

PULL_FILE = "SMD_pullx.xvg"
HBOND_FILE = "Hbond_Water.xvg"

OUTPUT_PNG = "Water_Hbond_vs_PullCoordinate.png"
OUTPUT_PDF = "Water_Hbond_vs_PullCoordinate.pdf"
OUTPUT_REPORT = "Water_Hbond_Statistics.txt"

NUMBER_OF_BINS = 60


# =====================================================
# READ XVG FILE
# =====================================================

def read_xvg(filename):

    data = []

    with open(filename, "r") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith(("@", "#")):
                continue

            values = np.fromstring(line, sep=" ")

            if len(values) >= 2:
                data.append(values[:2])

    if not data:
        raise ValueError(f"No numerical data found in {filename}")

    return np.asarray(data)


# =====================================================
# LOAD PULL COORDINATE
# =====================================================

print("Loading pull coordinate...")

pull_data = read_xvg(PULL_FILE)

pull_time_ps = pull_data[:, 0]
pull_coordinate = pull_data[:, 1]


# =====================================================
# LOAD WATER HBOND DATA
# =====================================================

print("Loading water H-bond data...")

hbond_data = read_xvg(HBOND_FILE)

hbond_time_ns = hbond_data[:, 0]
hbond_count = hbond_data[:, 1]

target_time_ps = hbond_time_ns * 1000.0


# =====================================================
# MATCH TIME POINTS
# =====================================================

indices = np.searchsorted(pull_time_ps, target_time_ps)

indices = np.clip(indices, 0, len(pull_time_ps) - 1)

previous_indices = np.maximum(indices - 1, 0)

current_error = np.abs(
    pull_time_ps[indices] - target_time_ps
)

previous_error = np.abs(
    pull_time_ps[previous_indices] - target_time_ps
)

use_previous = previous_error < current_error

indices[use_previous] = previous_indices[use_previous]

time_errors = np.abs(
    pull_time_ps[indices] - target_time_ps
)

if np.max(time_errors) > 0.0251:
    raise ValueError(
        f"Time mismatch detected: {np.max(time_errors):.5f} ps"
    )

z_coordinate = pull_coordinate[indices]


# =====================================================
# CALCULATE BINNED AVERAGE
# =====================================================

bin_edges = np.linspace(
    np.min(z_coordinate),
    np.max(z_coordinate),
    NUMBER_OF_BINS + 1
)

bin_indices = np.digitize(z_coordinate, bin_edges) - 1

bin_indices = np.clip(
    bin_indices,
    0,
    NUMBER_OF_BINS - 1
)

bin_x = []
bin_y = []

for i in range(NUMBER_OF_BINS):

    mask = bin_indices == i

    if np.any(mask):

        bin_x.append(np.mean(z_coordinate[mask]))
        bin_y.append(np.mean(hbond_count[mask]))

bin_x = np.asarray(bin_x)
bin_y = np.asarray(bin_y)


# =====================================================
# GENERATE GRAPH
# =====================================================

print("Generating Water H-bond graph...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    bin_x,
    bin_y,
    linewidth=2,
    label="Water–Peptide H-bonds"
)

ax.set_xlabel(
    "Relative Z Coordinate (nm)",
    fontsize=13
)

ax.set_ylabel(
    "Average Number of Hydrogen Bonds",
    fontsize=13
)

ax.set_title(
    "Water–Peptide H-bonds vs Relative Z Coordinate",
    fontsize=14
)

ax.legend(frameon=False)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# STATISTICS
# =====================================================

with open(OUTPUT_REPORT, "w") as file:

    file.write("WATER–PEPTIDE H-BOND ANALYSIS\n")
    file.write("Force Field: CHARMM36m\n\n")

    file.write(f"Total frames: {len(hbond_count)}\n")
    file.write(f"Minimum H-bonds: {np.min(hbond_count)}\n")
    file.write(f"Maximum H-bonds: {np.max(hbond_count)}\n")
    file.write(f"Mean H-bonds: {np.mean(hbond_count):.3f}\n")

    file.write(
        f"Non-zero frames: {np.count_nonzero(hbond_count)}\n"
    )

    file.write(
        f"Non-zero percentage: "
        f"{np.mean(hbond_count > 0) * 100:.2f}%\n"
    )

    file.write(
        f"Relative Z coordinate range: "
        f"{np.min(z_coordinate):.4f} to "
        f"{np.max(z_coordinate):.4f} nm\n"
    )


print("\nAnalysis completed successfully.")

print(f"PNG: {OUTPUT_PNG}")
print(f"PDF: {OUTPUT_PDF}")
print(f"Report: {OUTPUT_REPORT}")
