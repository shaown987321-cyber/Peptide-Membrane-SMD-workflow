
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# FILE SETTINGS
# ============================================================

PULL_FILE = "SMD_pullx.xvg"

HBOND_FILES = {
    "LPS Core": "Hbond_LPS-core.xvg",
    "Lipid A": "Hbond_Lipid-A.xvg",
    "Inner Leaflet": "Hbond_Inner.xvg"
}

OUTPUT_PNG = "Hbond_vs_PullCoordinate.png"
OUTPUT_PDF = "Hbond_vs_PullCoordinate.pdf"
OUTPUT_REPORT = "Hbond_Analysis_Report.txt"

# Number of bins used for averaging
NUMBER_OF_BINS = 60

# ============================================================
# READ XVG FILE
# ============================================================

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


# ============================================================
# LOAD PULL COORDINATE
# ============================================================

print("Loading SMD pull coordinate...")

pull_data = read_xvg(PULL_FILE)

pull_time_ps = pull_data[:, 0]
pull_coordinate = pull_data[:, 1]

print(f"Pull data points: {len(pull_time_ps)}")
print(f"Pull time range: {pull_time_ps[0]} to {pull_time_ps[-1]} ps")


# ============================================================
# MATCH HBOND TIME WITH PULL COORDINATE
# ============================================================

def match_pull_coordinate(hbond_time_ns):

    target_time_ps = hbond_time_ns * 1000.0

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

    errors = np.abs(
        pull_time_ps[indices] - target_time_ps
    )

    if np.max(errors) > 0.0251:
        raise ValueError(
            "H-bond and pull coordinate time points do not align. "
            f"Maximum mismatch: {np.max(errors):.5f} ps"
        )

    return pull_coordinate[indices]


# ============================================================
# CALCULATE BINNED AVERAGE
# ============================================================

def calculate_binned_average(x, y, number_of_bins=60):

    bin_edges = np.linspace(
        np.min(x),
        np.max(x),
        number_of_bins + 1
    )

    bin_indices = np.digitize(x, bin_edges) - 1

    bin_indices = np.clip(
        bin_indices,
        0,
        number_of_bins - 1
    )

    bin_x = []
    bin_y = []

    for i in range(number_of_bins):

        mask = bin_indices == i

        if np.any(mask):

            bin_x.append(np.mean(x[mask]))
            bin_y.append(np.mean(y[mask]))

    return np.asarray(bin_x), np.asarray(bin_y)


# ============================================================
# LOAD HBOND DATA AND PREPARE GRAPH
# ============================================================

print("\nProcessing H-bond data...")

plot_data = {}
report_lines = []

for label, filename in HBOND_FILES.items():

    print(f"Processing {label}...")

    data = read_xvg(filename)

    hbond_time_ns = data[:, 0]
    hbond_count = data[:, 1]

    # Match each H-bond frame with its pull coordinate
    x = match_pull_coordinate(hbond_time_ns)
    y = hbond_count

    # Calculate average H-bond count in coordinate bins
    bin_x, bin_y = calculate_binned_average(
        x,
        y,
        NUMBER_OF_BINS
    )

    plot_data[label] = (bin_x, bin_y)

    # ========================================================
    # STATISTICS
    # ========================================================

    minimum = np.min(y)
    maximum = np.max(y)
    mean = np.mean(y)

    nonzero_frames = np.count_nonzero(y)
    total_frames = len(y)

    nonzero_percentage = (
        nonzero_frames / total_frames
    ) * 100

    min_index = np.argmin(y)
    max_index = np.argmax(y)

    report_lines.append(f"\n{'=' * 55}")
    report_lines.append(f"{label}")
    report_lines.append(f"{'=' * 55}")

    report_lines.append(f"Total frames: {total_frames}")
    report_lines.append(f"Minimum H-bonds: {minimum}")
    report_lines.append(f"Maximum H-bonds: {maximum}")
    report_lines.append(f"Mean H-bonds: {mean:.3f}")

    report_lines.append(
        f"Non-zero frames: {nonzero_frames}/{total_frames}"
    )

    report_lines.append(
        f"Non-zero frames (%): {nonzero_percentage:.2f}%"
    )

    report_lines.append(
        f"Minimum at: {hbond_time_ns[min_index]:.3f} ns"
    )

    report_lines.append(
        f"Maximum at: {hbond_time_ns[max_index]:.3f} ns"
    )

    # ========================================================
    # CONTINUOUS ACTIVE INTERVALS
    # ========================================================

    active = y > 0

    changes = np.diff(
        np.concatenate(([0], active.astype(int), [0]))
    )

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    report_lines.append("\nContinuous H-bond active intervals:")

    intervals = []

    for start, end in zip(starts, ends):

        start_time = hbond_time_ns[start]
        end_time = hbond_time_ns[end]

        duration = (
            end_time - start_time
            + np.median(np.diff(hbond_time_ns))
        )

        coordinate_start = x[start]
        coordinate_end = x[end]

        coordinate_min = np.min(x[start:end + 1])
        coordinate_max = np.max(x[start:end + 1])

        intervals.append(duration)

        report_lines.append(
            f"{start_time:.3f} - {end_time:.3f} ns | "
            f"Duration: {duration:.3f} ns | "
            f"Coordinate: {coordinate_start:.3f} to "
            f"{coordinate_end:.3f} nm | "
            f"Coordinate range: {coordinate_min:.3f} to "
            f"{coordinate_max:.3f} nm"
        )

    if intervals:

        report_lines.append(
            f"\nLongest active interval: "
            f"{max(intervals):.3f} ns"
        )

    else:

        report_lines.append("No active intervals found.")


# ============================================================
# GENERATE LINE GRAPH
# ============================================================

print("\nGenerating line graph...")

fig, ax = plt.subplots(figsize=(10, 6))

for label, (bin_x, bin_y) in plot_data.items():

    ax.plot(
        bin_x,
        bin_y,
        linewidth=2.0,
        label=label
    )

ax.set_xlabel(
    "Pull COM Coordinate (nm)",
    fontsize=13
)

ax.set_ylabel(
    "Average Number of Hydrogen Bonds",
    fontsize=13
)

ax.set_title(
    "Hydrogen Bonds vs Pull COM Coordinate",
    fontsize=14
)

ax.legend(
    fontsize=11,
    frameon=False
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.tick_params(
    axis="both",
    labelsize=11
)

plt.tight_layout()

# Save high-resolution PNG
plt.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight"
)

# Save PDF
plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SAVE STATISTICAL REPORT
# ============================================================

with open(OUTPUT_REPORT, "w") as file:

    file.write(
        "HYDROGEN BOND ANALYSIS VS PULL COM COORDINATE\n"
    )

    file.write(
        "Force Field: CHARMM36m\n"
    )

    file.write(
        "SMD Simulation Duration: 60 ns\n"
    )

    file.write(
        f"Number of Coordinate Bins: {NUMBER_OF_BINS}\n"
    )

    file.write("\n".join(report_lines))

print("\nAnalysis completed successfully.")

print(f"PNG graph: {OUTPUT_PNG}")
print(f"PDF graph: {OUTPUT_PDF}")
print(f"Statistics: {OUTPUT_REPORT}")
