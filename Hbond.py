
import numpy as np
import matplotlib.pyplot as plt

files = {
    "LPS-core": "Hbond_LPS-core.xvg",
    "Lipid-A": "Hbond_Lipid-A.xvg",
    "Inner leaflet": "Hbond_Inner.xvg"
}

def analyze_hbonds(time, hbonds):
    minimum = np.min(hbonds)
    maximum = np.max(hbonds)
    mean = np.mean(hbonds)

    min_times = time[hbonds == minimum]
    max_times = time[hbonds == maximum]

    active = hbonds > 0
    changes = np.diff(np.r_[False, active, False].astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    intervals = []

    for start, end in zip(starts, ends):
        start_time = time[start]
        end_time = time[end - 1]
        duration = end_time - start_time
        intervals.append((start_time, end_time, duration))

    longest = max(intervals, key=lambda x: x[2]) if intervals else None

    return minimum, maximum, mean, min_times, max_times, intervals, longest


report = []

def log(message=""):
    print(message)
    report.append(message)


log("=" * 65)
log("       PEPTIDE–MEMBRANE HYDROGEN BOND ANALYSIS")
log("=" * 65)

plt.figure(figsize=(11, 6.5))

reference_time = None

for label, filename in files.items():

    data = np.loadtxt(filename, comments=("@", "#"))
    time = data[:, 0]
    hbonds = data[:, 1]

    if reference_time is None:
        reference_time = time
    elif not np.allclose(reference_time, time):
        raise ValueError(f"Time points do not match in {filename}")

    minimum, maximum, mean, min_times, max_times, intervals, longest = analyze_hbonds(time, hbonds)

    log(f"\nREGION: {label}")
    log("-" * 55)
    log(f"Minimum H-bonds       : {minimum:.0f}")
    log(f"Maximum H-bonds       : {maximum:.0f}")
    log(f"Mean H-bonds          : {mean:.3f}")
    log(f"Minimum observed at   : {min_times[0]:.3f} ns")
    log(f"Maximum observed at   : {max_times[0]:.3f} ns")
    log(f"Non-zero frames       : {np.sum(hbonds > 0)}/{len(hbonds)}")
    log(f"Active frames (%)    : {100*np.mean(hbonds > 0):.2f}%")

    log("\nContinuous H-bond presence intervals:")

    if intervals:
        for start, end, duration in intervals:
            log(f"  {start:.3f} - {end:.3f} ns (duration: {duration:.3f} ns)")
    else:
        log("  No H-bonds detected.")

    if longest:
        log(f"\nLongest active interval: {longest[0]:.3f} - {longest[1]:.3f} ns")
        log(f"Duration               : {longest[2]:.3f} ns")

    plt.plot(time, hbonds, linewidth=1.3, label=label)

log("\n" + "=" * 65)

plt.xlabel("Time (ns)", fontsize=13)
plt.ylabel("Number of Hydrogen Bonds", fontsize=13)
plt.title("Peptide–Membrane Hydrogen Bonds During SMD", fontsize=14)
plt.xlim(0, 60)
plt.ylim(bottom=0)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

plt.savefig("Hbond_Comparison.png", dpi=600, bbox_inches="tight")
plt.savefig("Hbond_Comparison.pdf", bbox_inches="tight")
plt.show()

with open("Hbond_Statistics.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")

print("\nStatistics saved to Hbond_Statistics.txt")
