#!/usr/bin/env python3
"""
Peptide-residue vs LPS-component contact heatmaps (publication style).

Expects files named  <RESIDUE>_<COMPONENT>_contacts.xvg  (e.g. ARG2_AKDO_contacts.xvg)
in <root>/contacts, and writes figures + tables to <root>/heatmaps.

Produces (each plot is a separate figure: PNG + PDF + CSV, nothing else)
  heatmap_mean_contacts.png/.pdf/.csv        mean contact count
  heatmap_contact_occupancy.png/.pdf/.csv    % of frames with >= 1 contact

Usage
  python plot_contact_heatmaps.py                       # defaults to ./TAT_LPS_contacts
  python plot_contact_heatmaps.py --root TAT_LPS_contacts --skip-ns 50 --time-unit ps
  python plot_contact_heatmaps.py --normalize           # per-atom (fill ATOM_COUNTS first)
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Preferred left-to-right order; any extra components found are appended alphabetically.
COMPONENT_ORDER = ["ECLIP", "AKDO", "AHEP", "AGLC", "AGAL"]

# Heavy-atom (or atom) count of each LPS component group, used only with --normalize.
# Fill these in from your index groups, e.g. {"ECLIP": 30, "AKDO": 18, ...}
ATOM_COUNTS = {}

FILE_RE = re.compile(r"^([A-Za-z]{3}\d+)_(.+)_contacts$")


# ----------------------------------------------------------------------------- I/O
def read_xvg(path, column=1):
    """Return (time, value) arrays from a GROMACS .xvg file (comments/legends skipped)."""
    t, v = [], []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#@&":
                continue
            parts = line.split()
            try:
                t.append(float(parts[0]))
                v.append(float(parts[column]))
            except (ValueError, IndexError):
                continue
    if not t:
        raise ValueError(f"no numeric data in {path.name}")
    t, v = np.asarray(t), np.asarray(v)
    ok = np.isfinite(t) & np.isfinite(v)
    return t[ok], v[ok]


def block_sem(x, n_blocks=5):
    """Standard error of the mean from block averages (accounts for autocorrelation)."""
    if len(x) < n_blocks * 2:
        return np.nan
    blocks = np.array_split(x, n_blocks)
    means = np.array([b.mean() for b in blocks])
    return means.std(ddof=1) / np.sqrt(n_blocks)


def discover(contact_dir):
    """Find residues/components from filenames; residues sorted by residue number."""
    pairs = {}
    for p in contact_dir.glob("*_contacts.xvg"):
        m = FILE_RE.match(p.stem)
        if m:
            pairs[(m.group(1).upper(), m.group(2))] = p
    if not pairs:
        sys.exit(f"No '*_contacts.xvg' files matching RESIDUE_COMPONENT in {contact_dir}")

    residues = sorted({r for r, _ in pairs}, key=lambda r: int(re.search(r"\d+", r).group()))
    found = {c for _, c in pairs}
    components = [c for c in COMPONENT_ORDER if c in found] + sorted(found - set(COMPONENT_ORDER))
    return residues, components, pairs


# ------------------------------------------------------------------------- plotting
def pick_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def draw_heatmap(ax, matrix, rows, cols, cmap, vmin, vmax, fmt, cbar_label, title, show_ylabels):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_box_aspect(1)  # square plot area regardless of number of rows/columns

    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows if show_ylabels else [])
    ax.set_xlabel("LPS component", labelpad=6)
    if show_ylabels:
        ax.set_ylabel("Peptide residue", labelpad=6)
    ax.set_title(title, pad=8, fontweight="bold")
    ax.tick_params(length=0, pad=5)

    # thin white cell borders
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # annotations: text colour chosen from the actual cell colour (works for any cmap)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            r, g, b, _ = im.cmap(im.norm(matrix[i, j]))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center",
                    fontsize=7.5, color="black" if lum > 0.55 else "white")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(cbar_label, labelpad=6)
    cbar.outline.set_linewidth(0.6)
    return im


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="TAT_LPS_contacts", help="folder containing contacts/")
    ap.add_argument("--column", type=int, default=1, help="xvg column with the contact count (0-based, default 1)")
    ap.add_argument("--time-unit", choices=["ps", "ns"], default="ps", help="time unit of column 0 in the xvg files")
    ap.add_argument("--skip-ns", type=float, default=0.0, help="equilibration time to discard (ns)")
    ap.add_argument("--normalize", action="store_true", help="divide counts by ATOM_COUNTS per component")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    root = Path(args.root)
    contact_dir, out_dir = root / "contacts", root / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    residues, components, pairs = discover(contact_dir)
    nr, nc = len(residues), len(components)

    if args.normalize:
        missing_atoms = [c for c in components if c not in ATOM_COUNTS]
        if missing_atoms:
            sys.exit(f"--normalize needs ATOM_COUNTS for: {', '.join(missing_atoms)}")

    to_ns = 1e-3 if args.time_unit == "ps" else 1.0
    mean = np.full((nr, nc), np.nan)
    sd = np.full((nr, nc), np.nan)
    sem = np.full((nr, nc), np.nan)
    occ = np.full((nr, nc), np.nan)
    nframes = np.zeros((nr, nc), dtype=int)
    problems = []

    for i, res in enumerate(residues):
        for j, comp in enumerate(components):
            path = pairs.get((res, comp))
            if path is None:
                problems.append(f"missing  {res}_{comp}_contacts.xvg")
                continue
            try:
                t, v = read_xvg(path, args.column)
            except ValueError as e:
                problems.append(f"invalid  {e}")
                continue
            keep = t * to_ns >= args.skip_ns
            v = v[keep]
            if v.size == 0:
                problems.append(f"empty after --skip-ns  {path.name}")
                continue
            if args.normalize:
                v = v / ATOM_COUNTS[comp]
            mean[i, j] = v.mean()
            sd[i, j] = v.std(ddof=1) if v.size > 1 else 0.0
            sem[i, j] = block_sem(v)
            occ[i, j] = 100.0 * np.mean(v > 0)
            nframes[i, j] = v.size

    if problems:
        print(f"ERROR: {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print("  ", p, file=sys.stderr)
        sys.exit("Heatmap not generated.")

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({
        "font.family": pick_font(), "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 11, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": args.dpi,
    })

    unit = "contacts per atom" if args.normalize else "contacts"
    plots = [
        ("heatmap_mean_contacts", mean, 0.0, max(float(mean.max()), 1e-9), ".2f",
         f"Mean contact count ({unit})", "Peptide–LPS Mean Contacts"),
        ("heatmap_contact_occupancy", occ, 0.0, 100.0, ".0f",
         "Contact occupancy (% of frames)", "Peptide–LPS Contact Occupancy"),
    ]

    saved = []
    for stem_name, mat, vmin, vmax, fmt, cbar_label, title in plots:
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)  # square figure
        draw_heatmap(ax, mat, residues, components, args.cmap, vmin, vmax, fmt, cbar_label, title, True)

        stem = out_dir / stem_name
        fig.savefig(f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

        # matching CSV: one row per residue, one column per component
        with open(f"{stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["residue"] + components)
            for i, res in enumerate(residues):
                w.writerow([res] + [f"{x:.6f}" for x in mat[i]])
        saved.append(stem_name)

    # ---------------------------------------------------------------- tables
    print(f"{nr} residues x {nc} components | discarded first {args.skip_ns} ns | {unit}")
    for name in saved:
        print(f"Saved: {out_dir / name}.png / .pdf / .csv")


if __name__ == "__main__":
    main()