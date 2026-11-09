#!/usr/bin/env python3
"""FEL of TAT-OL using Dz and contact count."""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter, uniform_filter1d

BASE = Path(__file__).resolve().parent
T, KT = 310.15, 0.0083145 * 310.15
FMAX, NBINS, SMOOTH = 10.0, 40, 1.0
CUTOFF_NM, AVG_NS = 0.35, 5.0
BASIN_CUT, N_PICK, MIN_SEP_NS, DPI = 1.0, 5, 30.0, 600

def load(name):
    path = BASE / name
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    d = np.loadtxt(path, comments=("#", "@"), ndmin=2)
    if d.shape[1] < 2 or len(d) < 2:
        raise ValueError(f"Invalid XVG data: {path}")
    return d[:, 0], d[:, 1]

t, dz = load("TAT-OL_Dz.xvg")
t2, cn = load("TAT-OL_contacts.xvg")
if len(t) != len(t2) or not np.allclose(t, t2):
    raise ValueError("Input time grids differ.")
if t.max() > 5000:
    t = t / 1000.0
dt = np.median(np.diff(t))
if dt <= 0 or not np.all(np.diff(t) > 0):
    raise ValueError("Time values must increase strictly.")

def pad(v):
    return v.min() - 0.1 * np.ptp(v), v.max() + 0.1 * np.ptp(v)

H, xe, ye = np.histogram2d(
    dz, cn, bins=NBINS,
    range=[pad(dz), (max(0, pad(cn)[0]), pad(cn)[1])])
H = gaussian_filter(H, SMOOTH) if SMOOTH > 0 else H
P = H / H.sum()
F = np.full_like(P, np.nan)
ok = P > 1e-6 * P.max()
F[ok] = -KT * np.log(P[ok])
F -= np.nanmin(F)
Fp = np.where(np.isnan(F), FMAX, np.minimum(F, FMAX))
xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2

ix = np.clip(np.digitize(dz, xe) - 1, 0, NBINS - 1)
iy = np.clip(np.digitize(cn, ye) - 1, 0, NBINS - 1)
F_frame = np.nan_to_num(F[ix, iy], nan=np.inf)
basin = F_frame <= BASIN_CUT
tb, xb, yb, fb = t[basin], dz[basin], cn[basin], F_frame[basin]
picked = []
if len(tb):
    sx, sy = max(xb.std(), 1e-12), max(yb.std(), 1e-12)
    dist = np.hypot((xb - np.median(xb)) / sx,
                    (yb - np.median(yb)) / sy)
    for k in np.argsort(dist):
        if all(abs(tb[k] - tb[p]) >= MIN_SEP_NS for p in picked):
            picked.append(k)
        if len(picked) == N_PICK:
            break
picked.sort(key=lambda k: tb[k])

# Save candidates with ΔG
np.savetxt(BASE / "smd_candidates.txt",
    np.column_stack((tb[picked] * 1000, xb[picked], yb[picked], fb[picked])),
    fmt="%.0f %.3f %.0f %.3f",
    header="time_ps  Dz_nm  contacts  dG_kJ_mol")

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.linewidth": 1.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "pdf.fonttype": 42, "savefig.facecolor": "white"})
cmap = mpl.colormaps["viridis"]

# Sky blue color for time series plots
SKY_BLUE = "#00BFFF"

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(BASE / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ==================== FEL Plot ====================
fig, ax = plt.subplots(figsize=(5.6, 4.6))
cf = ax.contourf(xc, yc, Fp.T, levels=np.linspace(0, FMAX, 51), cmap=cmap)
cf.set_edgecolor("face")
cl = ax.contour(xc, yc, Fp.T, levels=np.arange(2, FMAX, 2),
                colors="k", linewidths=0.5, alpha=0.5)
ax.clabel(cl, fmt="%d", fontsize=7)
ax.set(xlabel=r"$D_z$ (nm)",
       ylabel="TAT-OL contacts")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
cb = fig.colorbar(cf, ax=ax, ticks=np.arange(0, FMAX + 1, 2), pad=0.03)
cb.set_label(r"$\Delta G$ (kJ mol$^{-1}$)")
cb.outline.set_linewidth(1.0)
save(fig, "fig_FEL")

# ==================== Time series plots ====================
def timeseries(v, ylabel, name, integer=False):
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.plot(t, v, lw=0.5, color=SKY_BLUE, alpha=0.35, rasterized=True, label="raw")
    ax.plot(t, uniform_filter1d(v, max(1, round(AVG_NS / dt)), mode="nearest"),
            lw=1.6, color=SKY_BLUE, label=f"{AVG_NS:g} ns mean")
    ax.set(xlabel="Time (ns)", ylabel=ylabel, xlim=(t.min(), t.max()))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=integer))
    ax.legend(frameon=False, fontsize=8, ncol=2,
              loc="lower right", bbox_to_anchor=(1, 1), borderaxespad=0.2)
    save(fig, name)

timeseries(dz, r"$D_z$ (nm)", "fig_Dz")
timeseries(cn, "TAT-OL contacts", "fig_contacts_TAT-OL", True)

# ==================== Summary ====================
i0, j0 = np.unravel_index(np.nanargmin(F), F.shape)
print(f"{len(t)} frames, {t.min():.2f}-{t.max():.2f} ns")
print(f"FEL minimum: Dz={xc[i0]:.3f} nm, contacts={yc[j0]:.1f}")
print(f"Basin (<= {BASIN_CUT} kJ/mol): {basin.sum()} frames "
      f"({100 * basin.mean():.1f}%)")
print()
print("SMD candidates:")
print(f"{'time_ps':>10}  {'Dz_nm':>8}  {'contacts':>10}  {'dG_kJ_mol':>10}")
print("-" * 44)
for k in picked:
    print(f"{tb[k]*1000:10.0f}  {xb[k]:8.3f}  {yb[k]:10.0f}  {fb[k]:10.3f}")
if not picked:
    print("No suitable basin frames found.")