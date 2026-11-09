import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

KB = 0.0083145  # kJ mol^-1 K^-1
TEMPERATURE = 310.15  # K
BINS = 40


def load_xvg(filename):
    """Read valid time and projection columns from a GROMACS XVG file."""
    rows = []
    skipped = 0

    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith(("#", "@", "&")):
                continue

            fields = line.split()
            if len(fields) < 2:
                skipped += 1
                continue

            try:
                time, projection = float(fields[0]), float(fields[1])
            except ValueError:
                skipped += 1
                continue

            if np.isfinite(time) and np.isfinite(projection):
                rows.append((time, projection))
            else:
                skipped += 1

    if len(rows) < 2:
        raise ValueError(f"Not enough valid numeric data in {filename}")

    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc1", default="pc1.xvg")
    parser.add_argument("--pc2", default="pc2.xvg")
    parser.add_argument("-o", "--output", default="TAT_PCA_FEL")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--bins", type=int, default=BINS)
    args = parser.parse_args()

    if args.temperature <= 0 or args.bins < 2:
        raise ValueError("Temperature must be positive and bins must be >= 2.")

    # Resolve default input files relative to this script's directory.
    folder = Path(__file__).resolve().parent
    pc1_file = Path(args.pc1)
    pc2_file = Path(args.pc2)

    if not pc1_file.is_absolute():
        pc1_file = folder / pc1_file
    if not pc2_file.is_absolute():
        pc2_file = folder / pc2_file

    t1, pc1, skip1 = load_xvg(pc1_file)
    t2, pc2, skip2 = load_xvg(pc2_file)

    if len(t1) != len(t2) or not np.allclose(
        t1, t2, rtol=0, atol=1e-6
    ):
        raise ValueError(
            "PC1 and PC2 time columns do not match. "
            "Check pc1.xvg and pc2.xvg for missing or malformed rows."
        )

    if not np.all(np.diff(t1) >= 0):
        raise ValueError("Time values are not in ascending order.")

    hist, xedges, yedges = np.histogram2d(
        pc1, pc2, bins=args.bins
    )

    if hist.sum() == 0:
        raise ValueError("No valid observations were found in the histogram.")

    probability = hist / hist.sum()
    occupied = probability > 0
    free_energy = np.full(hist.shape, np.nan)

    free_energy[occupied] = (
        -KB * args.temperature
        * np.log(probability[occupied] / probability.max())
    )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    mesh = ax.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_invalid(free_energy.T),
        shading="auto",
        cmap="viridis",
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(r"Relative free energy (kJ mol$^{-1}$)")

    ax.set_xlabel("PC1 (nm)")
    ax.set_ylabel("PC2 (nm)")
    ax.tick_params(direction="out", width=0.8, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    output = Path(args.output)
    if not output.is_absolute():
        output = folder / output
    output.parent.mkdir(parents=True, exist_ok=True)

    png_file = output.with_suffix(".png")
    pdf_file = output.with_suffix(".pdf")

    fig.savefig(png_file, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    print("\n========== PCA FEL SUMMARY ==========")
    print(f"PC1 input             : {pc1_file}")
    print(f"PC2 input             : {pc2_file}")
    print(f"Valid aligned frames  : {len(pc1)}")
    print(f"Skipped lines PC1     : {skip1}")
    print(f"Skipped lines PC2     : {skip2}")
    print(f"Time range            : {t1.min():.3f} to {t1.max():.3f} ns")
    print(f"Temperature           : {args.temperature:.2f} K")
    print(f"Histogram bins        : {args.bins} x {args.bins}")
    print(f"Occupied bins         : {occupied.sum()}")
    print(f"Free-energy range     : 0 to {np.nanmax(free_energy):.3f} kJ/mol")
    print(f"PNG output            : {png_file}")
    print(f"PDF output            : {pdf_file}")


if __name__ == "__main__":
    main()