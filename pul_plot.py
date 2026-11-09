# =============================================================================
#   EASY XVG PLOTTER  (GROMACS .xvg  ->  PNG plot + Excel file + stats table)
#
#   HOW TO USE:  1) change the SETTINGS below   2) press Run.   That's all.
#   Do NOT edit anything below the line "DO NOT EDIT BELOW THIS LINE".
#
#   Put this script in the SAME folder as your .xvg files.
#   Text goes inside quotes "like this".  True / False must start with a capital.
# =============================================================================

# ----------------------------- 1. YOUR FILES ---------------------------------
# Write your file names here (in the order you want them plotted).
# Leave the list empty  FILES = []  to use ALL .xvg files in this folder.
FILES = [
    "SMD_pullf.xvg",
    "SMD_pullx.xvg",
]

# -------------------------- 2. WHAT KIND OF PLOT -----------------------------
# "separate"           -> SEPARATE picture for each file vs time (e.g. pullf vs time,
#                         pullx vs time) PLUS a picture of file 1 vs file 2 (force vs distance)
# "stacked"            -> one graph per file, one under the other (same time axis)
# "overlay"            -> all files drawn together in ONE graph (different colours)
# "force_vs_position"  -> needs exactly 2 files: 1st file on Y axis, 2nd on X axis
PLOT_STYLE = "separate"

# ------------------------------ 3. COLOURS & LINES ---------------------------
# Colour names ("blue", "red", "black", "green", "orange", "purple", "gray")
# or hex codes ("#1f77b4").
LINE_COLOR = "#1f77b4"            # colour of the line (stacked & force_vs_position)
OVERLAY_COLORS = ["blue", "red", "green", "orange", "purple"]   # used only for "overlay"
LINE_WIDTH = 1.0                  # thicker line = bigger number (try 0.5 to 2)

# --------------------------- 4. TITLES & AXIS NAMES --------------------------
# Leave "" (empty quotes) to use the names that are already inside the .xvg file.
MAIN_TITLE = ""                   # big title on top of the whole figure
PANEL_TITLES = []                 # one title per file, e.g. ["Pulling force", "COM position"]
X_LABEL = ""                      # X axis name, e.g. "Time (ns)"
Y_LABELS = []                     # one per file, e.g. ["Force (kJ/mol/nm)", "Distance (nm)"]

# ------------------------------- 5. AXIS RANGE -------------------------------
# Time is shown in ns. Use None for automatic, or numbers like (0, 30).
X_RANGE = None                    # e.g. (0, 30)  -> show only the first 30 ns
Y_RANGE = None                    # e.g. (0, 1200)   (ignored in "separate" style)
TIME_UNIT = "ns"                  # "ns", "ps" or "us"  (input .xvg time is assumed to be ps)

# ------------------------------- 6. EXTRA OPTIONS ----------------------------
HORIZONTAL_LINES = []             # draw dashed guide lines, e.g. [0, 500]
COLUMNS = None                    # None = all data columns; or e.g. [1, 3] = 1st and 3rd
SHOW_GRID = True
FONT_SIZE = 12
FIGURE_SIZE = None                # None = automatic; or (10, 6)  (width, height in inches)
DPI = 300                         # picture quality (300 = print quality)

# --------------------------------- 7. OUTPUT ---------------------------------
PICTURE_NAME = ""                 # "" = automatic (named after first file), or "my_plot.png"
                                  # (not used in "separate" style: names are made automatically)
MAKE_EXCEL = True                 # True = also save an Excel file (time in ns)
EXCEL_NAME = ""                   # "" = automatic, or "my_data.xlsx"
EXCEL_EVERY = 0                   # 0 = automatic. Excel max is 1,048,576 rows; long files
                                  # are thinned automatically. Or write e.g. 10 = every 10th point.
PRINT_STATS = True                # True = print mean / std / min / max / peak in the terminal
SHOW_WINDOW = False               # True = also open the plot in a window


# =============================================================================
#                      DO NOT EDIT BELOW THIS LINE
# =============================================================================
import glob
import os
import re
import sys

import numpy as np
import matplotlib

if not SHOW_WINDOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TIME_FACTORS = {"ps": 1.0, "ns": 1e-3, "us": 1e-6}
EXCEL_MAX_ROWS = 1_048_575
MAX_PLOT_POINTS = 20000        # only thins the picture, never the stats / Excel


def read_xvg(path):
    meta = {"title": "", "xlabel": "", "ylabel": "", "legends": {}}
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("@"):
                for key, pat in (("title", r'@\s+title\s+"(.*)"'),
                                 ("xlabel", r'@\s+xaxis\s+label\s+"(.*)"'),
                                 ("ylabel", r'@\s+yaxis\s+label\s+"(.*)"')):
                    m = re.match(pat, s)
                    if m:
                        meta[key] = m.group(1)
                m = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', s)
                if m:
                    meta["legends"][int(m.group(1))] = m.group(2)
                continue
            rows.append(s)
    if not rows:
        sys.exit(f"ERROR: no data found in {path}")
    data = np.loadtxt(rows, ndmin=2)
    meta["x"], meta["y"] = data[:, 0], data[:, 1:].T
    return meta


def nice(text):
    """Make xmgrace markup readable in plots."""
    text = re.sub(r"\\S(.*?)\\N", r"$^{\1}$", text)
    text = re.sub(r"\\s(.*?)\\N", r"$_{\1}$", text)
    return text.replace("\\xD\\f{}", "Δ").replace("\\f{}", "")


def plain(text):
    text = re.sub(r"\\[Ss](.*?)\\N", r"\1", text)
    return text.replace("\\xD\\f{}", "Δ").replace("\\f{}", "")


def name_of(meta, c, stem, as_plain=False):
    f = plain if as_plain else nice
    if c in meta["legends"]:
        return f(meta["legends"][c])
    if meta["y"].shape[0] == 1:
        return f(meta["title"]) or stem
    return f"{stem} col {c + 1}"


def thin(x, y):
    if len(x) <= MAX_PLOT_POINTS:
        return x, y
    step = int(np.ceil(len(x) / MAX_PLOT_POINTS))
    return x[::step], y[::step]


def pick(lst, i, default=""):
    return lst[i] if i < len(lst) and lst[i] else default


def columns_of(meta):
    n = meta["y"].shape[0]
    cols = [c - 1 for c in COLUMNS if 1 <= c <= n] if COLUMNS else list(range(n))
    if not cols:
        sys.exit(f"ERROR: COLUMNS setting is out of range (file has {n} data columns)")
    return cols


def print_stats(rows):
    head = f"{'Series':32s} {'mean':>11s} {'std':>10s} {'min':>11s} {'max':>11s} {'peak':>11s} {'@ ns':>8s}"
    print("\n" + head + "\n" + "-" * len(head))
    for name, t, y in rows:
        i = int(np.argmax(np.abs(y)))
        print(f"{name[:32]:32s} {y.mean():11.4g} {y.std():10.4g} {y.min():11.4g} "
              f"{y.max():11.4g} {y[i]:11.4g} {t[i]:8.3f}")
    print()


def get_xlsxwriter():
    """Import xlsxwriter; if it is missing, install it automatically (needs internet)."""
    try:
        import xlsxwriter
        return xlsxwriter
    except ImportError:
        import subprocess
        print("Excel helper 'xlsxwriter' is missing - installing it now ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"])
            import xlsxwriter
            return xlsxwriter
        except Exception:
            return None


def write_excel(path, metas, stems):
    xlsxwriter = get_xlsxwriter()
    if xlsxwriter is None:
        print("\nCould not install xlsxwriter, so NO Excel file was made.\n"
              "Open the PyCharm Terminal and run:   pip install xlsxwriter\n"
              "then run this script again. (The picture and stats above are fine.)")
        return None
    wb = xlsxwriter.Workbook(path, {"constant_memory": True})
    bold = wb.add_format({"bold": True})
    same_time = all(len(m["x"]) == len(metas[0]["x"]) and np.allclose(m["x"], metas[0]["x"])
                    for m in metas)

    def dump(ws, t_ns, columns):
        n = len(t_ns)
        step = EXCEL_EVERY if EXCEL_EVERY > 0 else int(np.ceil(n / EXCEL_MAX_ROWS))
        ws.write(0, 0, "Time (ns)", bold)
        for j, (name, _) in enumerate(columns, start=1):
            ws.write(0, j, name, bold)
        ws.set_column(0, len(columns), 22)
        ws.freeze_panes(1, 0)
        for r, i in enumerate(range(0, n, step), start=1):
            ws.write_number(r, 0, float(t_ns[i]))
            for j, (_, y) in enumerate(columns, start=1):
                ws.write_number(r, j, float(y[i]))
        return step

    if same_time:
        cols = []
        for m, stem in zip(metas, stems):
            unit = plain(m["ylabel"])
            for c in range(m["y"].shape[0]):
                nm = name_of(m, c, stem, True)
                cols.append((f"{nm} [{unit}]" if unit and unit not in nm else nm, m["y"][c]))
        step = dump(wb.add_worksheet("data"), metas[0]["x"] * 1e-3, cols)
    else:
        step = 1
        for m, stem in zip(metas, stems):
            cols = [(name_of(m, c, stem, True), m["y"][c]) for c in range(m["y"].shape[0])]
            step = max(step, dump(wb.add_worksheet(stem[:31]), m["x"] * 1e-3, cols))
    wb.close()
    return step


def plot_separate(files, metas, stems, factor):
    """One picture per file (vs time) + one picture file1-vs-file2. Returns stats rows."""
    stats, shown = [], []
    for i, (meta, stem) in enumerate(zip(metas, stems)):
        cols = columns_of(meta)
        fig, ax = plt.subplots(figsize=FIGURE_SIZE or (9, 4.5))
        for k, c in enumerate(cols):
            color = LINE_COLOR if len(cols) == 1 else OVERLAY_COLORS[k % len(OVERLAY_COLORS)]
            ax.plot(*thin(meta["x"] * factor, meta["y"][c]), color=color, lw=LINE_WIDTH,
                    label=name_of(meta, c, stem))
            stats.append((name_of(meta, c, stem, True), meta["x"] * 1e-3, meta["y"][c]))
        if len(cols) > 1:
            ax.legend(frameon=False)
        ax.set_xlabel(X_LABEL or f"Time ({TIME_UNIT})")
        ax.set_ylabel(pick(Y_LABELS, i, nice(meta["ylabel"])))
        ax.set_title(pick(PANEL_TITLES, i, meta["title"]))
        for h in HORIZONTAL_LINES:
            ax.axhline(h, color="gray", ls="--", lw=0.8)
        if X_RANGE:
            ax.set_xlim(*X_RANGE)
        fig.tight_layout()
        out = os.path.splitext(files[i])[0] + "_vs_time.png"
        fig.savefig(out, dpi=DPI)
        print(f"Saved picture: {out}")
        if SHOW_WINDOW:
            shown.append(fig)
        else:
            plt.close(fig)
    if len(metas) >= 2:
        m1, m2 = metas[0], metas[1]
        n = min(len(m1["x"]), len(m2["x"]))
        c1, c2 = columns_of(m1)[0], columns_of(m2)[0]
        fig, ax = plt.subplots(figsize=FIGURE_SIZE or (7, 5))
        ax.plot(*thin(m2["y"][c2][:n], m1["y"][c1][:n]), color=LINE_COLOR, lw=LINE_WIDTH)
        ax.set_xlabel(pick(Y_LABELS, 1, nice(m2["ylabel"])))
        ax.set_ylabel(pick(Y_LABELS, 0, nice(m1["ylabel"])))
        ax.set_title(MAIN_TITLE or f"{m1['title']} vs {m2['title']}")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(files[0]), f"{stems[0]}_vs_{stems[1]}.png")
        fig.savefig(out, dpi=DPI)
        print(f"Saved picture: {out}")
    return stats


def main():
    if TIME_UNIT not in TIME_FACTORS:
        sys.exit('ERROR: TIME_UNIT must be "ns", "ps" or "us"')
    if PLOT_STYLE not in ("separate", "stacked", "overlay", "force_vs_position"):
        sys.exit('ERROR: PLOT_STYLE must be "separate", "stacked", "overlay" or "force_vs_position"')

    files = [f if os.path.isabs(f) else os.path.join(HERE, f) for f in FILES]
    if not files:
        files = sorted(glob.glob(os.path.join(HERE, "*.xvg")))
    missing = [f for f in files if not os.path.exists(f)]
    if missing or not files:
        sys.exit("ERROR: file(s) not found:\n  " + "\n  ".join(missing or [HERE + "  (no .xvg files here)"]) +
                 "\nPut the .xvg files in the same folder as this script, or fix FILES.")
    print("Using files:\n  " + "\n  ".join(files))

    metas = [read_xvg(f) for f in files]
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    base = os.path.splitext(files[0])[0]
    factor = TIME_FACTORS[TIME_UNIT]

    plt.rcParams.update({"font.size": FONT_SIZE, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": SHOW_GRID, "grid.alpha": 0.25})
    stats = []
    fig = None

    if PLOT_STYLE == "separate":
        stats = plot_separate(files, metas, stems, factor)
    elif PLOT_STYLE == "force_vs_position":
        if len(metas) != 2:
            sys.exit('ERROR: "force_vs_position" needs exactly 2 files in FILES')
        m1, m2 = metas
        n = min(len(m1["x"]), len(m2["x"]))
        c1, c2 = columns_of(m1)[0], columns_of(m2)[0]
        yv, xv = m1["y"][c1][:n], m2["y"][c2][:n]
        fig, ax = plt.subplots(figsize=FIGURE_SIZE or (7, 5))
        ax.plot(*thin(xv, yv), color=LINE_COLOR, lw=LINE_WIDTH)
        ax.set_xlabel(X_LABEL or nice(m2["ylabel"]))
        ax.set_ylabel(pick(Y_LABELS, 0, nice(m1["ylabel"])))
        ax.set_title(MAIN_TITLE or f"{m1['title']} vs {m2['title']}")
        axes = [ax]
        t = m1["x"][:n] * 1e-3
        stats += [(name_of(m1, c1, stems[0], True), t, yv), (name_of(m2, c2, stems[1], True), t, xv)]
    else:
        overlay = PLOT_STYLE == "overlay"
        if overlay:
            fig, ax = plt.subplots(figsize=FIGURE_SIZE or (9, 5))
            axes = [ax] * len(metas)
        else:
            fig, axs = plt.subplots(len(metas), 1, sharex=True,
                                    figsize=FIGURE_SIZE or (9, 3.2 * len(metas) + 0.8))
            axes = np.atleast_1d(axs).tolist()
        k = 0
        for i, (ax, meta, stem) in enumerate(zip(axes, metas, stems)):
            cols = columns_of(meta)
            for c in cols:
                color = OVERLAY_COLORS[k % len(OVERLAY_COLORS)] if overlay else LINE_COLOR
                ax.plot(*thin(meta["x"] * factor, meta["y"][c]), color=color, lw=LINE_WIDTH,
                        label=name_of(meta, c, stem) if overlay else None)
                stats.append((name_of(meta, c, stem, True), meta["x"] * 1e-3, meta["y"][c]))
                k += 1
            if overlay:
                if i == 0:
                    ax.set_ylabel(pick(Y_LABELS, 0, nice(meta["ylabel"])))
            else:
                ax.set_ylabel(pick(Y_LABELS, i, nice(meta["ylabel"])))
                ax.set_title(pick(PANEL_TITLES, i, meta["title"]), fontsize=FONT_SIZE, loc="left")
            for h in HORIZONTAL_LINES:
                ax.axhline(h, color="gray", ls="--", lw=0.8)
        axes[-1].set_xlabel(X_LABEL or f"Time ({TIME_UNIT})")
        if overlay:
            axes[0].set_title(MAIN_TITLE or metas[0]["title"])
            axes[0].legend(frameon=False)
        elif MAIN_TITLE:
            fig.suptitle(MAIN_TITLE)

    if fig is not None:
        for ax in set(axes):
            if X_RANGE:
                ax.set_xlim(*X_RANGE)
            if Y_RANGE:
                ax.set_ylim(*Y_RANGE)
        fig.tight_layout()
        out = PICTURE_NAME or base + ".png"
        if not os.path.isabs(out):
            out = os.path.join(HERE, out) if PICTURE_NAME else out
        fig.savefig(out, dpi=DPI)
        print(f"\nSaved picture: {out}")

    if PRINT_STATS:
        print_stats(stats)

    if MAKE_EXCEL:
        xl = EXCEL_NAME or base + ".xlsx"
        if EXCEL_NAME and not os.path.isabs(xl):
            xl = os.path.join(HERE, xl)
        step = write_excel(xl, metas, stems)
        if step is None:
            return
        note = f"  (every {step}th point, because of Excel's row limit)" if step > 1 and EXCEL_EVERY == 0 else ""
        print(f"Saved Excel  : {xl}{note}")

    if SHOW_WINDOW:
        plt.show()


main()