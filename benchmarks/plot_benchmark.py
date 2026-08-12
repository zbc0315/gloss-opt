"""Generate publication figures for the 5 benchmark studies.

Output: benchmarks/plots/bench_fig{1-5}.jpg

Usage:
    python -m benchmarks.plot_benchmark
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Arial"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Shared styles ─────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "GLOSS": "#2166ac",
    "UCB-BO": "#b2182b",
    "BO(EI)": "#d6604d",
    "GA": "#4dac26",
    "Random": "#969696",
}
ALGO_LINES = {
    "GLOSS": "-",
    "UCB-BO": "--",
    "BO(EI)": "-.",
    "GA": ":",
    "Random": ":",
}
ALGO_MARKERS = {
    "GLOSS": "o",
    "UCB-BO": "s",
    "BO(EI)": "^",
    "GA": "D",
    "Random": "v",
}
ALGO_ORDER = ["GLOSS", "UCB-BO", "BO(EI)", "GA", "Random"]

# Display names for legends/titles ONLY (data-lookup keys stay "UCB-BO"/"BO(EI)"
# to match the result CSVs). Matches the renamed terminology in the paper.
DISPLAY_NAME = {"UCB-BO": "BO-UCB", "BO(EI)": "BO-EI"}
def _disp(a):
    return DISPLAY_NAME.get(a, a)

RATIO_COLORS = {"GLOSS(2:3:3)": "#d62728", "GLOSS(4:2:2)": "#2166ac", "GLOSS(6:1:1)": "#2ca02c"}
RATIO_LINES = {"GLOSS(2:3:3)": "-", "GLOSS(4:2:2)": "--", "GLOSS(6:1:1)": "-."}
RATIO_MARKERS = {"GLOSS(2:3:3)": "o", "GLOSS(4:2:2)": "s", "GLOSS(6:1:1)": "D"}
RATIO_ORDER = ["GLOSS(2:3:3)", "GLOSS(4:2:2)", "GLOSS(6:1:1)"]


def _convergence(ax, df, dataset, algos, colors, lines, show_legend=True,
                  show_ylabel=True, fs=None, xmax=None):
    """Plot mean ± std convergence curves."""
    fs_label = 9 if fs is None else fs
    fs_tick = 8 if fs is None else fs
    fs_legend = 7 if fs is None else fs
    sub = df[df["dataset"] == dataset]
    rounds = sorted(sub["round"].unique())
    disp = [r + 1 for r in rounds]          # init(-1)->0, rec rounds 0..19 -> 1..20
    for algo in algos:
        asub = sub[sub["algorithm"] == algo]
        if len(asub) == 0:
            continue
        means = [asub[asub["round"] == r]["pct_opt"].mean() for r in rounds]
        stds = [asub[asub["round"] == r]["pct_opt"].std() for r in rounds]
        means, stds = np.array(means), np.array(stds)
        ax.plot(disp, means, color=colors[algo], ls=lines[algo], lw=1.8, label=_disp(algo))
        ax.fill_between(disp, means - stds, means + stds, alpha=0.12, color=colors[algo])
    ax.axhline(95, color="black", ls="--", lw=1.6, alpha=0.6,
               label="95% of optimum")
    ax.set_xlabel("Round", fontsize=fs_label)
    if show_ylabel:
        ax.set_ylabel("% of optimum", fontsize=fs_label)
    if xmax is None:
        xmax = 20
    ax.set_xlim(0, xmax)
    ax.tick_params(labelsize=fs_tick)
    if show_legend:
        ax.legend(fontsize=fs_legend, loc="lower right", framealpha=0.7)


def _t95_bar(ax, data, algos, colors, ylabel="t95 (rounds)"):
    """Bar chart of mean t95."""
    x = np.arange(len(data))
    width = 0.8 / len(algos)
    for i, algo in enumerate(algos):
        vals = [d.get(algo, 20) for d in data]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
                      color=colors[algo], label=algo, edgecolor="white", linewidth=0.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Study 1 — Main comparison (3 datasets × 5 algos)
# ═══════════════════════════════════════════════════════════════════════════════

# Cache signature for Fig 1 raw-data .npz. Bump when upstream loader params change.
_FIG1_CACHE_SIG = "v2:bh_tsne+qm9_100k_gap+arr_10k_p3_w0.08_s42"


def _load_main_fig_data():
    """Load + cache raw X/y plus precomputed BH t-SNE for Fig 1 rows 1/2.

    Cache invalidates if signature differs (loader params changed).
    """
    cache = os.path.join(RESULTS_DIR, "bench_main_fig_data.npz")
    if os.path.exists(cache):
        try:
            data = np.load(cache, allow_pickle=False)
            if str(data["signature"]) == _FIG1_CACHE_SIG:
                return {
                    "BH": {"X": data["bh_X"], "y": data["bh_y"], "embed": data["bh_tsne"]},
                    "QM9": {"X": data["qm_X"], "y": data["qm_y"]},
                    "AR": {"X": data["ar_X"], "y": data["ar_y"]},
                }
        except (KeyError, ValueError):
            pass
        print("  Cache signature mismatch — regenerating.")

    from benchmarks.datasets import load_qm9, load_buchwald_hartwig
    from benchmarks.virtual_functions import get_virtual_dataset
    from sklearn.manifold import TSNE

    print("  Loading Buchwald-Hartwig...")
    bh = load_buchwald_hartwig()
    print("  Computing BH t-SNE (one-hot features collapse under PCA)...")
    bh_tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                   init="pca", learning_rate="auto").fit_transform(bh["candidates"])
    print("  Loading QM9-gap (100k, this takes a while)...")
    qm = load_qm9(n_samples=100000, target="gap")
    print("  Loading Arrhenius-2D...")
    ar = get_virtual_dataset("arrhenius", n_candidates=10000, n_peaks=3,
                              noise_sigma=0.0, peak_width=0.08, seed=42)

    np.savez(cache,
             signature=np.array(_FIG1_CACHE_SIG),
             bh_X=bh["candidates"], bh_y=bh["y_true"], bh_tsne=bh_tsne,
             qm_X=qm["candidates"], qm_y=qm["y_true"],
             ar_X=ar["candidates"], ar_y=ar["y_true"])
    return {
        "BH": {"X": bh["candidates"], "y": bh["y_true"], "embed": bh_tsne},
        "QM9": {"X": qm["candidates"], "y": qm["y_true"]},
        "AR": {"X": ar["candidates"], "y": ar["y_true"]},
    }


def _add_panel_label(ax, letter, on_dark=False):
    if on_dark:
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", color="white",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
    else:
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))


def _row1_structure(ax, item, mode, label_letter, title, xlabel_pair,
                    cmap_name, cbar_title, fs=None):
    """Row 1 panel.

    mode:
      'pca_global'        — PCA fit on full X, transform 4000-point subsample (QM9)
      'tsne_precomputed'  — scatter of cached t-SNE embedding (BH)
      'function_heatmap'  — evaluate Arrhenius on a 200×200 grid, imshow
    """
    on_dark = False
    if mode == "function_heatmap":
        from benchmarks.virtual_functions import arrhenius_yield
        xs = np.linspace(0, 1, 200)
        XX, YY = np.meshgrid(xs, xs)
        grid = np.column_stack([XX.ravel(), YY.ravel()])
        Z = arrhenius_yield(grid, n_peaks=3, peak_width=0.08,
                            noise_sigma=0.0, seed=42).reshape(XX.shape)
        im = ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower",
                       cmap=cmap_name, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        on_dark = True  # heatmap dark-end dominates — white label on dark box
    else:
        rng = np.random.RandomState(42)
        X, y = item["X"], item["y"]
        if mode == "pca_global":
            from sklearn.decomposition import PCA
            n = min(4000, len(X))
            idx = rng.choice(len(X), n, replace=False)
            XY = PCA(n_components=2, random_state=42).fit(X).transform(X[idx])
            yp = y[idx]
        elif mode == "tsne_precomputed":
            XY, yp = item["embed"], y
        else:
            raise ValueError(f"Unknown mode: {mode}")
        sc = ax.scatter(XY[:, 0], XY[:, 1], c=yp, s=6, cmap=cmap_name,
                        alpha=0.75, edgecolors="none")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)

    fs_label = 9 if fs is None else fs
    fs_title = 10 if fs is None else fs
    fs_tick = 8 if fs is None else fs
    fs_cbar_title = 8 if fs is None else fs
    fs_cbar_tick = 7 if fs is None else fs
    cbar.ax.tick_params(labelsize=fs_cbar_tick)
    cbar.ax.set_title(cbar_title, fontsize=fs_cbar_title, pad=4)
    ax.set_xlabel(xlabel_pair[0], fontsize=fs_label)
    ax.set_ylabel(xlabel_pair[1], fontsize=fs_label)
    ax.set_title(title, fontsize=fs_title, fontweight="bold")
    ax.tick_params(labelsize=fs_tick)
    _add_panel_label(ax, label_letter, on_dark=on_dark)


def _row2_distribution(ax, y, label_letter, xlabel, cmap_name,
                        show_ylabel=False, mode="hist", fs=None, ylabel=None):
    """Row 2: y-value distribution colored by value using row-1's cmap.

    mode='hist' → histogram with bars filled at bin-center color (D, E).
    mode='kde'  → continuous density curve with gradient fill (F).

    Annotation in upper-right shows the percentage of candidates above the
    95 % threshold (top-95 % share), the same threshold marked by the red
    dashed line.
    """
    y = np.asarray(y, dtype=float)
    y_min, y_opt = float(y.min()), float(y.max())
    rng_y = (y_opt - y_min) if y_opt > y_min else 1.0
    thresh = y_min + 0.95 * rng_y
    n_top = int((y >= thresh).sum())
    top_pct = n_top / len(y) * 100

    cmap = plt.get_cmap(cmap_name)

    if mode == "kde":
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(y)
        x_grid = np.linspace(y_min, y_opt, 400)
        d = kde(x_grid)
        # Gradient fill under the KDE curve, slice by slice.
        for i in range(len(x_grid) - 1):
            color = cmap((x_grid[i] - y_min) / rng_y)
            ax.fill_between(x_grid[i:i+2], 0, d[i:i+2],
                            color=color, edgecolor="none")
        ax.plot(x_grid, d, color="black", lw=0.6)
        ax.set_ylim(bottom=0)
    else:
        counts, edges, patches = ax.hist(y, bins=50, edgecolor="black", linewidth=0.3)
        centers = (edges[:-1] + edges[1:]) / 2
        for center, pat in zip(centers, patches):
            pat.set_facecolor(cmap((center - y_min) / rng_y))

    fs_label = 9 if fs is None else fs
    fs_tick = 8 if fs is None else fs
    fs_text = 8 if fs is None else fs
    ax.axvline(thresh, color="red", ls="--", lw=1.2)
    ax.set_xlabel(xlabel, fontsize=fs_label)
    if show_ylabel:
        ax.set_ylabel(ylabel or ("Density" if mode == "kde" else "Count"),
                      fontsize=fs_label)
    ax.tick_params(labelsize=fs_tick)
    ax.text(0.97, 0.97, f"95% of optimum: {top_pct:.3f}%", transform=ax.transAxes,
            fontsize=fs_text, ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", pad=3))
    _add_panel_label(ax, label_letter)


def fig1_main():
    df_conv = pd.read_csv(os.path.join(RESULTS_DIR, "bench_main_summary.csv"))
    raw = _load_main_fig_data()

    cols = [
        {"key": "BH", "csv_name": "Buchwald-Hartwig",
         "title": "Buchwald-Hartwig (3955, Exp.)",
         "xlabel": "Yield (%)", "structure": "tsne_precomputed",
         "axes_labels": ("t-SNE 1", "t-SNE 2"),
         "cmap": "viridis", "cbar_title": "Yield (%)",
         "dist_mode": "hist"},
        {"key": "QM9", "csv_name": "QM9-gap",
         "title": "QM9-gap (100k, DFT)",
         "xlabel": "HOMO-LUMO gap (Hartree)", "structure": "pca_global",
         "axes_labels": ("PC1", "PC2"),
         "cmap": "magma", "cbar_title": "Gap (Hartree)",
         "dist_mode": "hist"},
        {"key": "AR", "csv_name": "Arrhenius-2D",
         "title": "Arrhenius-2D\n(Continuous distribution function)",
         "xlabel": r"$f(x)$", "structure": "function_heatmap",
         "axes_labels": (r"$x_1$", r"$x_2$"),
         "cmap": "Spectral_r", "cbar_title": r"$f(x)$",
         "dist_mode": "kde"},
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    LETTERS = "ABCDEFGHI"
    FS = 12  # uniform font size for all text in this figure (panel letters
             #  A-I keep their distinct size set in _add_panel_label).

    # Row 1: structure (A, B, C)
    for i, c in enumerate(cols):
        _row1_structure(axes[0, i], raw[c["key"]], c["structure"],
                        LETTERS[i], c["title"], c["axes_labels"],
                        c["cmap"], c["cbar_title"], fs=FS)

    # Row 2: y distribution (D, E, F) — D/E hist; F KDE for continuous Arrhenius
    row2_ylabels = ["Count", "Count", "Density"]
    for i, c in enumerate(cols):
        _row2_distribution(axes[1, i], raw[c["key"]]["y"],
                           LETTERS[3 + i], c["xlabel"], c["cmap"],
                           show_ylabel=(row2_ylabels[i] is not None),
                           mode=c["dist_mode"], fs=FS, ylabel=row2_ylabels[i])

    # Row 3: convergence (G, H, I) — legend on G, y-ticks shown on all panels
    for i, c in enumerate(cols):
        ax = axes[2, i]
        _convergence(ax, df_conv, c["csv_name"], ALGO_ORDER, ALGO_COLORS,
                     ALGO_LINES, show_legend=(i == 0), show_ylabel=(i == 0),
                     fs=FS)
        _add_panel_label(ax, LETTERS[6 + i])

    fig.tight_layout(w_pad=2.0, h_pad=2.5)
    out = os.path.join(PLOTS_DIR, "bench_fig1_main.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight", format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Study 2 — Scaling (t95 vs n + convergence at selected scales)
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_scaling():
    """Panel A: integer y-axis with a // break and a single 'failed' y-tick
    for (algo, n) means that hit the budget. Lines stay continuous across
    the break.
    """
    from matplotlib.ticker import FixedLocator

    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_scaling_summary.csv"))
    n_rounds = int(df["round"].max()) + 1

    def extract_n(name):
        return int(name.replace("QM9-gap-", "").replace("k", ""))

    all_ds = sorted(df["dataset"].unique(), key=extract_n)
    ns = [extract_n(d) for d in all_ds]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    FAILED_Y = n_rounds + 5      # 1-indexed rounds: max real t95 is 20, failed above
    BREAK_MID = n_rounds + 3     # midpoint of the 20 <-> failed gap
    BREAK_LO = BREAK_MID - 0.7
    BREAK_HI = BREAK_MID + 0.7
    Y_TOP    = n_rounds + 7

    # Panel A
    ax = axes[0, 0]
    for algo in ALGO_ORDER:
        t95s = []
        for ds in all_ds:
            sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo)]
            seeds_t95 = []
            for seed in sub["seed"].unique():
                ssub = sub[sub["seed"] == seed]
                reached = ssub[ssub["pct_opt"] >= 95]
                seeds_t95.append(int(reached["round"].min()) + 1
                                 if len(reached) > 0 else n_rounds + 1)
            t95s.append(float(np.mean(seeds_t95)))

        plot_y = [FAILED_Y if t >= n_rounds + 1 else t for t in t95s]
        ax.plot(ns, plot_y, color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                marker=ALGO_MARKERS[algo], ms=5, lw=1.5, label=_disp(algo))

    ax.set_ylim(0, Y_TOP)
    ticks = [0, 4, 8, 12, 16, 20, FAILED_Y]
    labels = ["0", "4", "8", "12", "16", "20", "failed"]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels(labels)

    trans = ax.get_yaxis_transform()
    ax.plot([0, 0], [BREAK_LO, BREAK_HI], color="white", linewidth=3,
            transform=trans, clip_on=False, zorder=10)
    slash_kw = dict(color="black", linewidth=1.0,
                    transform=trans, clip_on=False, zorder=11)
    dx, dy = 0.012, 0.233
    y_mid = BREAK_MID
    ax.plot([-dx, +dx], [y_mid - dy - 0.1, y_mid + dy - 0.1], **slash_kw)
    ax.plot([-dx, +dx], [y_mid - dy + 0.1, y_mid + dy + 0.1], **slash_kw)

    ax.set_xlabel("n (×1000)", fontsize=10)
    ax.set_ylabel("$t_{95}$ (rounds)", fontsize=10)
    ax.set_title(r"$\mathbf{t_{95}}$ vs Space Size", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.text(0.02, 0.97, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # Panels B/C/D: convergence at n=20k, 50k, 80k. The legend is added
    # manually on D (below) with markers so it also documents Panel A's
    # per-algorithm marker shapes; the convergence lines stay marker-free.
    panel_specs = [
        (0, 1, 20, "B", True),
        (1, 0, 50, "C", True),
        (1, 1, 80, "D", False),
    ]
    for row, col, n, label, show_yl in panel_specs:
        ax = axes[row, col]
        ds = f"QM9-gap-{n}k"
        _convergence(ax, df, ds, ALGO_ORDER, ALGO_COLORS, ALGO_LINES,
                     show_legend=False, show_ylabel=show_yl, fs=10)
        ax.set_xticks([0, 5, 10, 15, 20])   # integer Round ticks
        ax.set_ylim(30, 110)                 # unified y-range for B/C/D
        ax.set_title(f"n = {n}k", fontsize=10, fontweight="bold")
        ax.text(0.02, 0.97, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        if not show_yl:
            ax.tick_params(left=False, labelleft=False)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=ALGO_COLORS[a], ls=ALGO_LINES[a],
               marker=ALGO_MARKERS[a], ms=5, lw=1.5, label=_disp(a))
        for a in ALGO_ORDER
    ]
    legend_handles.append(
        Line2D([0], [0], color="black", ls="--", lw=1.6, alpha=0.6,
               label="95% of optimum")
    )
    axes[1, 1].legend(handles=legend_handles, fontsize=10,
                      loc="lower right", framealpha=0.7)

    fig.tight_layout(w_pad=1.5, h_pad=2.0)
    out = os.path.join(PLOTS_DIR, "bench_fig2_scaling.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 (stratified): Scaling on the *quantile-stratified* QM9 subsets.
# Reads bench_scaling_strat_summary.csv; writes bench_fig2_scaling_strat.jpg
# (does NOT overwrite the original).
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_scaling_strat():
    """Panel A: integer t95 on the y-axis with a // break above the budget
    cap and one extra y-tick labelled "failed" beyond the break. Mean t95
    that hit the budget (i.e. every seed exhausted the round limit) is
    plotted at the "failed" tick; lines stay continuous through the break.
    """
    from matplotlib.ticker import FixedLocator

    csv = os.path.join(RESULTS_DIR, "bench_scaling_strat_summary.csv")
    if not os.path.exists(csv):
        raise SystemExit(
            f"missing {csv}; run `python -m benchmarks.bench_scaling_strat` first")
    df = pd.read_csv(csv)
    n_rounds = int(df["round"].max()) + 1

    def extract_n(name):
        return int(name.replace("QM9-gap-strat-", "").replace("k", ""))

    all_ds = sorted(df["dataset"].unique(), key=extract_n)
    ns = [extract_n(d) for d in all_ds]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # y-axis layout for Panel A
    FAILED_Y = n_rounds + 2          # sentinel position for "all seeds failed"
    BREAK_LO = n_rounds + 0.4        # gap (//) covers this band on the spine
    BREAK_HI = n_rounds + 1.4
    Y_TOP    = n_rounds + 3

    # --- Panel A: lines stay continuous across the break ---
    ax = axes[0, 0]
    for algo in ALGO_ORDER:
        t95s = []
        for ds in all_ds:
            sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo)]
            seeds_t95 = []
            for seed in sub["seed"].unique():
                ssub = sub[sub["seed"] == seed]
                reached = ssub[ssub["pct_opt"] >= 95]
                seeds_t95.append(int(reached["round"].min())
                                 if len(reached) > 0 else n_rounds)
            t95s.append(float(np.mean(seeds_t95)))

        # Mean = n_rounds → "all 5 seeds failed" → plot at FAILED_Y; else as-is.
        plot_y = [FAILED_Y if t >= n_rounds else t for t in t95s]
        ax.plot(ns, plot_y, color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                marker=ALGO_MARKERS[algo], ms=5, lw=1.5, label=_disp(algo))

    ax.set_ylim(0, Y_TOP)
    ticks = [0, 4, 8, 12, 16, 20, FAILED_Y]
    labels = ["0", "4", "8", "12", "16", "20", "failed"]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels(labels)

    # White "cut" on the y-spine to mark the break, plus two // diagonals.
    trans = ax.get_yaxis_transform()  # x in axis-frac, y in data
    ax.plot([0, 0], [BREAK_LO, BREAK_HI], color="white", linewidth=3,
            transform=trans, clip_on=False, zorder=10)
    slash_kw = dict(color="black", linewidth=1.0,
                    transform=trans, clip_on=False, zorder=11)
    dx, dy = 0.012, 0.233
    y_mid = (BREAK_LO + BREAK_HI) / 2
    ax.plot([-dx, +dx], [y_mid - dy, y_mid + dy], **slash_kw)
    ax.plot([-dx, +dx], [y_mid - dy + 0.2, y_mid + dy + 0.2], **slash_kw)

    ax.set_xlabel("n (×1000)", fontsize=9)
    ax.set_ylabel("t95 (rounds)", fontsize=9)
    ax.set_title("t95 vs Space Size", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.97, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # --- Panels B / C / D: convergence curves at n = 20k, 50k, 80k ---
    for row, col, n, label, show_yl, show_lg in [
        (0, 1, 20, "B", True, False),
        (1, 0, 50, "C", True, False),
        (1, 1, 80, "D", False, True),
    ]:
        ax = axes[row, col]
        ds = f"QM9-gap-strat-{n}k"
        _convergence(ax, df, ds, ALGO_ORDER, ALGO_COLORS, ALGO_LINES,
                     show_legend=show_lg, show_ylabel=show_yl,
                     xmax=n_rounds)
        ax.set_title(f"n = {n}k", fontsize=10, fontweight="bold")
        ax.text(0.02, 0.97, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        if not show_yl:
            ax.tick_params(left=False, labelleft=False)

    fig.tight_layout(w_pad=1.5, h_pad=2.0)
    out_local = os.path.join(PLOTS_DIR, "bench_fig2_scaling_strat.jpg")
    fig.savefig(out_local, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out_local}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 (v100): Scaling at n_rounds = 100. Reads bench_scaling_v100_summary.csv.
# Output: bench_fig2_scaling_v100.jpg (does NOT overwrite the original).
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_scaling_v100():
    csv = os.path.join(RESULTS_DIR, "bench_scaling_v100_summary.csv")
    if not os.path.exists(csv):
        raise SystemExit(
            f"missing {csv}; run `python -m benchmarks.bench_scaling_v100` first")
    df = pd.read_csv(csv)
    n_rounds = int(df["round"].max()) + 1  # rounds are 0-indexed up to N-1

    def extract_n(name):
        return int(name.replace("QM9-gap-", "").replace("k", ""))

    all_ds = sorted(df["dataset"].unique(), key=extract_n)
    ns = [extract_n(d) for d in all_ds]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Panel A: t95 vs n (budget = n_rounds; never-reached → t95 = budget)
    ax = axes[0, 0]
    for algo in ALGO_ORDER:
        t95s = []
        for ds in all_ds:
            sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo)]
            seeds_t95 = []
            for seed in sub["seed"].unique():
                ssub = sub[sub["seed"] == seed]
                reached = ssub[ssub["pct_opt"] >= 95]
                seeds_t95.append(int(reached["round"].min())
                                 if len(reached) > 0 else n_rounds)
            t95s.append(np.mean(seeds_t95))
        ax.plot(ns, t95s, color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                marker=ALGO_MARKERS[algo], ms=4, lw=1.5, label=algo)
    ax.set_xlabel("n (×1000)", fontsize=9)
    ax.set_ylabel(f"t95 (rounds; budget = {n_rounds})", fontsize=9)
    ax.set_title(f"t95 vs Space Size (budget {n_rounds})",
                 fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.97, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # Panels B/C/D: convergence curves at n=20k, 50k, 80k, full horizon
    panel_specs = [
        (0, 1, 20, "B", True, False),
        (1, 0, 50, "C", True, False),
        (1, 1, 80, "D", False, True),
    ]
    for row, col, n, label, show_yl, show_lg in panel_specs:
        ax = axes[row, col]
        ds = f"QM9-gap-{n}k"
        _convergence(ax, df, ds, ALGO_ORDER, ALGO_COLORS, ALGO_LINES,
                     show_legend=show_lg, show_ylabel=show_yl,
                     xmax=n_rounds)
        ax.set_title(f"n = {n}k  (budget {n_rounds})",
                     fontsize=10, fontweight="bold")
        ax.text(0.02, 0.97, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        if not show_yl:
            ax.tick_params(left=False, labelleft=False)

    fig.tight_layout(w_pad=1.5, h_pad=2.0)
    out = os.path.join(PLOTS_DIR, "bench_fig2_scaling_v100.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Study 3 — Complexity (t95 bar + convergence)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_complexity():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_complexity_summary.csv"))
    levels = ["C1-simple", "C2-moderate", "C3-multi", "C4-complex", "C5-extreme"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Panel A: t95 bar chart
    ax = axes[0]
    t95_data = []
    for lv in levels:
        row = {}
        for algo in ALGO_ORDER:
            sub = df[(df["dataset"] == lv) & (df["algorithm"] == algo)]
            if len(sub) == 0:
                row[algo] = 20
                continue
            seeds_t95 = []
            for seed in sub["seed"].unique():
                ssub = sub[sub["seed"] == seed]
                reached = ssub[ssub["pct_opt"] >= 95]
                seeds_t95.append(int(reached["round"].min()) if len(reached) > 0 else 20)
            row[algo] = np.mean(seeds_t95)
        t95_data.append(row)

    x = np.arange(len(levels))
    width = 0.15
    for i, algo in enumerate(ALGO_ORDER):
        vals = [d[algo] for d in t95_data]
        ax.bar(x + i * width - 0.3, vals, width * 0.9,
               color=ALGO_COLORS[algo], label=algo, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["C1", "C2", "C3", "C4", "C5"], fontsize=8)
    ax.set_ylabel("t95 (rounds)", fontsize=9)
    ax.set_title("t95 vs Complexity", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.97, "A", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Panels B-D: convergence at C1, C3, C5 (legend on C)
    for i, lv in enumerate(["C1-simple", "C3-multi", "C5-extreme"]):
        ax = axes[i + 1]
        _convergence(ax, df, lv, ALGO_ORDER, ALGO_COLORS, ALGO_LINES,
                     show_legend=(i == 1), show_ylabel=(i == 0))
        label = lv.split("-")[0]
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.text(0.02, 0.97, chr(66 + i), transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        if i > 0:
            ax.tick_params(left=False, labelleft=False)

    fig.tight_layout(w_pad=1.5)
    out = os.path.join(PLOTS_DIR, "bench_fig3_complexity.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight", format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Study 4 — Ratio × Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_ratio_scaling():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_ratio_scaling_summary.csv"))

    def extract_n(name):
        return int(name.replace("QM9-gap-", "").replace("k", ""))

    all_ds = sorted(df["dataset"].unique(), key=extract_n)
    ns = [extract_n(d) for d in all_ds]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Panel A: t95 vs n
    ax = axes[0]
    for algo in RATIO_ORDER:
        t95s = []
        for ds in all_ds:
            sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo)]
            seeds_t95 = []
            for seed in sub["seed"].unique():
                ssub = sub[sub["seed"] == seed]
                reached = ssub[ssub["pct_opt"] >= 95]
                seeds_t95.append(int(reached["round"].min()) if len(reached) > 0 else 20)
            t95s.append(np.mean(seeds_t95))
        ax.plot(ns, t95s, color=RATIO_COLORS[algo], ls=RATIO_LINES[algo],
                marker=RATIO_MARKERS[algo], ms=4, lw=1.5, label=algo)
    ax.set_xlabel("n (×1000)", fontsize=9)
    ax.set_ylabel("t95 (rounds)", fontsize=9)
    ax.set_title("t95 vs Space Size", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.97, "A", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Panels B-D: convergence at n=10k, 50k, 100k
    for i, n in enumerate([10, 50, 100]):
        ax = axes[i + 1]
        ds = f"QM9-gap-{n}k"
        _convergence(ax, df, ds, RATIO_ORDER, RATIO_COLORS, RATIO_LINES,
                     show_legend=False, show_ylabel=(i == 0))
        ax.set_title(f"n = {n}k", fontsize=10, fontweight="bold")
        ax.text(0.02, 0.97, chr(66 + i), transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        if i > 0:
            ax.tick_params(left=False, labelleft=False)

    fig.tight_layout(w_pad=1.5)
    out = os.path.join(PLOTS_DIR, "bench_fig4_ratio_scaling.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight", format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Study 5 — Ratio × nD Complexity
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_ratio_complexity():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_ratio_complexity_summary.csv"))
    levels = sorted(df["dataset"].unique(), key=lambda x: int(x.split("-")[0].replace("RC", "")))
    dim_labels = {"RC1-2D-3C": "2D", "RC2-4D-5C": "4D", "RC3-6D-5C": "6D",
                  "RC4-8D-7C": "8D", "RC5-10D-7C": "10D"}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Panel A: t95 bar chart by dimensionality
    ax = axes[0]
    t95_data = []
    for lv in levels:
        row = {}
        for algo in RATIO_ORDER:
            sub = df[(df["dataset"] == lv) & (df["algorithm"] == algo)]
            if len(sub) == 0:
                row[algo] = 20
                continue
            seeds_t95 = []
            for seed in sub["seed"].unique():
                ssub = sub[sub["seed"] == seed]
                reached = ssub[ssub["pct_opt"] >= 95]
                seeds_t95.append(int(reached["round"].min()) if len(reached) > 0 else 20)
            row[algo] = np.mean(seeds_t95)
        t95_data.append(row)

    x = np.arange(len(levels))
    width = 0.25
    for i, algo in enumerate(RATIO_ORDER):
        vals = [d[algo] for d in t95_data]
        ax.bar(x + i * width - 0.25, vals, width * 0.9,
               color=RATIO_COLORS[algo], label=algo, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([dim_labels.get(l, l) for l in levels], fontsize=8)
    ax.set_ylabel("t95 (rounds)", fontsize=9)
    ax.set_xlabel("Dimensionality", fontsize=9)
    ax.set_title("t95 vs Complexity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.97, "A", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Panels B-D: convergence at 2D, 6D, 10D
    for i, lv in enumerate(["RC1-2D-3C", "RC3-6D-5C", "RC5-10D-7C"]):
        ax = axes[i + 1]
        _convergence(ax, df, lv, RATIO_ORDER, RATIO_COLORS, RATIO_LINES,
                     show_legend=False, show_ylabel=(i == 0))
        ax.set_title(dim_labels.get(lv, lv), fontsize=10, fontweight="bold")
        ax.text(0.02, 0.97, chr(66 + i), transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        if i > 0:
            ax.tick_params(left=False, labelleft=False)

    fig.tight_layout(w_pad=1.5)
    out = os.path.join(PLOTS_DIR, "bench_fig5_ratio_complexity.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight", format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Trajectory analysis on Arrhenius-2D (4 algorithms, single seed)
# ═══════════════════════════════════════════════════════════════════════════════

def _split_main_points_blocks():
    """bench_main_points.csv has no dataset/algo/seed columns. Reconstruct
    blocks by init-row boundaries; the loop order in study_main is fixed:
      datasets = [QM9-gap, Buchwald-Hartwig, Arrhenius-2D]  (outer)
      seeds    = [47, 53, 59, 61, 67]                       (middle)
      algos    = [GLOSS, UCB-BO, BO(EI), GA, Random]        (inner)
    → 3 × 5 × 5 = 75 blocks, each starts where round=-1 & point_idx=0.
    """
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_main_points.csv"))
    starts = df[(df["round"] == -1) & (df["point_idx"] == 0)].index.tolist()
    assert len(starts) == 75, f"expected 75 blocks, got {len(starts)}"
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(df)
        blocks.append(df.iloc[s:e].reset_index(drop=True))
    return blocks


def _heatmap_text_color(tx, ty, Z, vmin, vmax, cmap):
    """Pick black/white text color based on local heatmap luminance."""
    ny, nx = Z.shape
    i = int(np.clip(tx * nx, 0, nx - 1))
    j = int(np.clip(ty * ny, 0, ny - 1))
    rgb = cmap((Z[j, i] - vmin) / (vmax - vmin))[:3]
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if lum > 0.55 else "white"


def _pick_label_pos(px, py, obstacles, near_d=0.045, far_d=0.085, safe=0.038):
    """Find a label position near (px, py) that clears `obstacles` (data coords).

    Returns (tx, ty, needs_arrow). Tries `near_d` first; if every direction
    collides, falls back to `far_d` and signals the caller to draw a leader line.
    """
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    def best_at(d):
        cands = []
        for a in angles:
            tx, ty = px + d * np.cos(a), py + d * np.sin(a)
            if not (0.04 <= tx <= 0.96 and 0.04 <= ty <= 0.96):
                continue
            min_d = (min(np.hypot(tx - ox, ty - oy) for ox, oy in obstacles)
                     if obstacles else 1.0)
            cands.append((min_d, tx, ty))
        return max(cands, key=lambda c: c[0]) if cands else None

    near = best_at(near_d)
    if near is not None and near[0] >= safe:
        _, tx, ty = near
        return tx, ty, False
    far = best_at(far_d) or near
    if far is None:
        return px, py + 0.05, True
    _, tx, ty = far
    return tx, ty, True


def _draw_trajectory(ax, df_algo, algo_name, x0g, x1g, Z, y_opt, y_min):
    from matplotlib.lines import Line2D

    ax.pcolormesh(x0g, x1g, Z, cmap="Spectral_r", shading="auto", rasterized=True)
    zmin, zmax = float(Z.min()), float(Z.max())
    heat_cmap = plt.get_cmap("Spectral_r")

    init_rows = df_algo[df_algo["round"] == -1]
    all_rounds = df_algo[df_algo["round"] >= 0]
    ib = init_rows.loc[init_rows["y"].idxmax()]
    ax.scatter(ib["x0"], ib["x1"], marker="P", s=100, color="white",
               edgecolors="black", linewidths=0.8, zorder=5)

    # Per-round running best
    bpr = []
    running = float(ib["y"])
    for r in sorted(all_rounds["round"].unique()):
        rs = all_rounds[all_rounds["round"] == r]
        rec = rs.loc[rs["y"].idxmax()]
        if rec["y"] > running:
            running = float(rec["y"])
        bpr.append(rec)
    bpr = pd.DataFrame(bpr).reset_index(drop=True)

    # Truncate at the first round whose running-best rounds to 100 % at the
    # title's display precision (matches the {:.0f}% used in set_title below).
    pct = (np.maximum.accumulate(np.concatenate([[ib["y"]], bpr["y"].to_numpy()]))[1:]
           - y_min) / (y_opt - y_min) * 100
    reached = np.where(pct >= 99.5)[0]
    if len(reached) > 0:
        bpr = bpr.iloc[: reached[0] + 1].reset_index(drop=True)
        pct = pct[: reached[0] + 1]

    lx = [ib["x0"]] + bpr["x0"].tolist()
    ly = [ib["x1"]] + bpr["x1"].tolist()
    ax.plot(lx, ly, color="black", lw=1.2, zorder=2, alpha=0.85)

    ALGO_MARKER = {"GLOSS": "o", "UCB-BO": "s", "BO(EI)": "^", "GA": "D"}
    mk = ALGO_MARKER.get(algo_name, "o")

    final_reached_100 = len(bpr) > 0 and float(pct[-1]) >= 99.5
    n_plot = len(bpr) - 1 if final_reached_100 else len(bpr)
    if n_plot > 0:
        ax.scatter(bpr["x0"].iloc[:n_plot], bpr["x1"].iloc[:n_plot],
                   marker=mk, s=55, color="white", alpha=0.95,
                   edgecolors="black", linewidths=0.5, zorder=3)
    if final_reached_100:
        last = bpr.iloc[-1]
        ax.scatter(last["x0"], last["x1"], marker=mk, s=55,
                   color="yellow", alpha=0.95,
                   edgecolors="black", linewidths=0.6, zorder=4)

    # Round-number labels: round 1, round 2, the round at which running best
    # first hit its final value, plus algo-specific extras.
    EXTRA_ROUNDS = {"GA": [14]}  # 1-indexed round 15
    # Manual overrides keyed by (algo_name, 1-indexed round).
    # 'pos': absolute (x, y) in data coords; 'arrow': force a leader line.
    LABEL_OVERRIDES = {
        ("GA",      2): {"arrow": True},
        ("GLOSS",   3): {"text": "3-10"},

        # Panel B (UCB-BO): consolidate 3-5 and 15-20
        ("UCB-BO",  3): {"text": "3-5"},
        ("UCB-BO",  4): {"skip": True},
        ("UCB-BO",  5): {"skip": True},
        ("UCB-BO", 13): {"skip": True},
        ("UCB-BO", 14): {"skip": True},
        ("UCB-BO", 15): {"text": "15-20",
                         "pos": (0.88, 0.50), "arrow": True},
        ("UCB-BO", 16): {"skip": True},
        ("UCB-BO", 17): {"skip": True},
        ("UCB-BO", 18): {"skip": True},
        ("UCB-BO", 19): {"skip": True},
        ("UCB-BO", 20): {"skip": True},

        # Panel C (BO(EI)): consolidate 3-5, "10,12-14", "15-17,19,20"
        ("BO(EI)",  3): {"text": "3-5"},
        ("BO(EI)",  4): {"skip": True},
        ("BO(EI)",  5): {"skip": True},
        ("BO(EI)",  7): {"skip": True},
        ("BO(EI)", 10): {"skip": True},
        ("BO(EI)", 12): {"skip": True},
        ("BO(EI)", 13): {"text": "10,12-14"},
        ("BO(EI)", 14): {"skip": True},
        ("BO(EI)", 15): {"text": "15-17,19,20",
                         "pos": (0.88, 0.55), "arrow": True},
        ("BO(EI)", 16): {"skip": True},
        ("BO(EI)", 17): {"skip": True},
        ("BO(EI)", 19): {"skip": True},
        ("BO(EI)", 20): {"skip": True},
    }
    if len(bpr) > 0:
        running = np.maximum.accumulate(bpr["y"].to_numpy())
        peak_idx = int(np.argmax(running))
        if algo_name in ("UCB-BO", "BO(EI)", "GA"):
            wanted = set(range(len(bpr)))
        else:
            wanted = {0, 1, 2, peak_idx} | set(EXTRA_ROUNDS.get(algo_name, []))
        annotate_idx = sorted(i for i in wanted if 0 <= i < len(bpr))

        # All scatter points are obstacles; we'll exclude the labelled point itself.
        scatter_pts = [(float(ib["x0"]), float(ib["x1"]))]
        scatter_pts += [(float(r["x0"]), float(r["x1"])) for _, r in bpr.iterrows()]

        # Place "initial" with same algorithm so it doesn't collide either.
        init_obs = [p for p in scatter_pts if p != (float(ib["x0"]), float(ib["x1"]))]
        ix, iy, init_arrow = _pick_label_pos(
            float(ib["x0"]), float(ib["x1"]), init_obs)
        ic = _heatmap_text_color(ix, iy, Z, zmin, zmax, heat_cmap)
        ax.annotate(
            "initial", xy=(float(ib["x0"]), float(ib["x1"])),
            xytext=(ix, iy), xycoords="data", textcoords="data",
            fontsize=7, fontweight="bold", color=ic, zorder=7,
            ha="center", va="center",
            arrowprops=(dict(arrowstyle="-", lw=0.8, color="black",
                             linestyle=(0, (3, 2)),
                             shrinkA=2, shrinkB=3)
                        if init_arrow else None))
        # 'initial' label position becomes an obstacle for round labels.
        scatter_pts.append((ix, iy))

        for idx in annotate_idx:
            rec = bpr.iloc[idx]
            px, py = float(rec["x0"]), float(rec["x1"])
            obs = [p for p in scatter_pts if p != (px, py)]
            override = LABEL_OVERRIDES.get((algo_name, idx + 1))
            if override and override.get("skip"):
                continue
            if override and "pos" in override:
                tx, ty = override["pos"]
                needs_arrow = True
            else:
                tx, ty, needs_arrow = _pick_label_pos(px, py, obs)
                if override and override.get("arrow"):
                    needs_arrow = True
            tc = _heatmap_text_color(tx, ty, Z, zmin, zmax, heat_cmap)
            label_text = (override.get("text", str(idx + 1))
                          if override else str(idx + 1))
            ax.annotate(
                label_text, xy=(px, py), xytext=(tx, ty),
                xycoords="data", textcoords="data",
                fontsize=8, fontweight="bold", color=tc, zorder=7,
                ha="center", va="center",
                arrowprops=(dict(arrowstyle="-", lw=0.8, color="black",
                                 linestyle=(0, (3, 2)),
                                 shrinkA=1, shrinkB=2)
                            if needs_arrow else None))
            # Now this label is itself an obstacle for later labels.
            scatter_pts.append((tx, ty))

    final_pct = float(pct[len(bpr) - 1]) if len(bpr) > 0 else 0.0
    title_peak_round = (peak_idx + 1) if len(bpr) > 0 else 0
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x_1$", fontsize=10)
    ax.set_ylabel(r"$x_2$", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_title(f"{_disp(algo_name)} ({final_pct:.0f}% optimum by Round {title_peak_round})",
                 fontsize=10, fontweight="bold", pad=4)

    handles = [
        Line2D([0], [0], marker="P", color="w", markerfacecolor="white",
               markersize=9, markeredgecolor="black", label="init best", lw=0),
        Line2D([0], [0], marker=mk, color="w", markerfacecolor="white",
               markersize=8, markeredgecolor="black",
               label=f"{_disp(algo_name)} per-round best", lw=0),
    ]
    if final_reached_100:
        handles.append(
            Line2D([0], [0], marker=mk, color="w", markerfacecolor="yellow",
                   markersize=8, markeredgecolor="black",
                   label="reached 100%", lw=0))
    ax.legend(handles=handles, loc="lower left", fontsize=10,
              framealpha=0.85, edgecolor="#cccccc")


def fig6_trajectory():
    """Arrhenius-2D trajectories for 4 algorithms, single seed (2×2)."""
    from benchmarks.virtual_functions import arrhenius_yield

    SEEDS = [47, 53, 59, 61, 67]
    ALGOS_ALL = ["GLOSS", "UCB-BO", "BO(EI)", "GA", "Random"]
    SEED_PICK = 61
    ALGOS_PLOT = ["GLOSS", "GA", "BO(EI)", "UCB-BO"]   # B/D slots swapped

    blocks = _split_main_points_blocks()
    AR_START = 50  # Arrhenius is the 3rd dataset → blocks 50–74
    s_idx = SEEDS.index(SEED_PICK)
    panels = {a: blocks[AR_START + s_idx * 5 + ALGOS_ALL.index(a)]
              for a in ALGOS_PLOT}

    xs = np.linspace(0, 1, 300)
    XX, YY = np.meshgrid(xs, xs)
    Z = arrhenius_yield(np.column_stack([XX.ravel(), YY.ravel()]),
                        n_peaks=3, peak_width=0.08, noise_sigma=0.0,
                        seed=42).reshape(XX.shape)
    # Use the same y_opt the benchmark uses (max over the 10k candidate pool)
    # so subplot percentages match bench_main_summary.csv exactly.
    ar_y = _load_main_fig_data()["AR"]["y"]
    y_opt = float(ar_y.max())
    y_min = float(ar_y.min())

    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    LETTERS = "ABCD"
    for idx, algo in enumerate(ALGOS_PLOT):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        _draw_trajectory(ax, panels[algo], algo, xs, xs, Z, y_opt, y_min)
        ax.text(0.02, 0.97, LETTERS[idx], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                color="white")

    fig.tight_layout(rect=[0, 0, 0.92, 1.0])

    # Shared f(x) colorbar to the right of panel D, matching D's height,
    # placed in reserved right-margin space so the four subplots stay equal
    # and aligned.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    posD = axes[1, 1].get_position()
    cax = fig.add_axes([posD.x1 + 0.02, posD.y0, 0.018, posD.height])
    sm = ScalarMappable(
        norm=Normalize(vmin=float(Z.min()), vmax=float(Z.max())),
        cmap="Spectral_r")
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$f(x)$", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    out = os.path.join(PLOTS_DIR, "bench_fig6_trajectory.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Alternative Figure 3: one subplot per algorithm, box plots of t95 (5 seeds)
# versus chemical space size. 2x3 grid (5 algorithms + 1 note cell). Writes a
# SEPARATE file and does NOT overwrite the current Figure 3 (bench_fig2_scaling).
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_scaling_boxplot(algos=None, nrows=2, ncols=3,
                         out_name="bench_scaling_boxplot.jpg"):
    if algos is None:
        algos = ALGO_ORDER
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_scaling_summary.csv"))
    n_rounds = int(df["round"].max()) + 1          # 20 recommendation rounds

    def extract_n(name):
        return int(name.replace("QM9-gap-", "").replace("k", ""))

    all_ds = sorted(df["dataset"].unique(), key=extract_n)
    ns = [extract_n(d) for d in all_ds]
    FAIL = 25                                       # "failed" row, set above the reached t95 range (1-20)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.25 * nrows),
                             sharey=True)
    axflat = np.atleast_1d(axes).flatten()

    for a_idx, algo in enumerate(algos):
        ax = axflat[a_idx]
        data = []
        for ds in all_ds:
            sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo)]
            vals = []
            for seed in sub["seed"].unique():
                ssub = sub[sub["seed"] == seed]
                reached = ssub[ssub["pct_opt"] >= 95]
                vals.append(int(reached["round"].min()) + 1
                            if len(reached) > 0 else FAIL)
            data.append(vals)

        color = ALGO_COLORS[algo]
        bp = ax.boxplot(data, positions=ns, widths=3.2, patch_artist=True,
                        manage_ticks=False,
                        medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker="o", ms=3, alpha=0.5,
                                        markerfacecolor=color,
                                        markeredgecolor=color))
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.55, edgecolor=color)
        for part in bp["whiskers"] + bp["caps"]:
            part.set(color=color)

        ax.axhline(FAIL, color="gray", ls=":", lw=0.9, alpha=0.7)
        ax.set_title(_disp(algo), fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel(r"Chemical space size $n$ ($\times 1000$)", fontsize=12)
        if a_idx % ncols == 0:
            ax.set_ylabel(r"$t_{95}$ (rounds)", fontsize=12)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, FAIL + 3)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 5, 10, 15, 20, FAIL])
        ax.set_yticklabels(["0", "5", "10", "15", "20", "failed"])
        ax.tick_params(labelsize=12)

    # blank any unused cells
    for j in range(len(algos), len(axflat)):
        axflat[j].axis("off")

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Bar-chart variant of the scaling figure: same per-algorithm subplot structure
# as fig3_scaling_boxplot, but showing the mean t95 vs chemical space size (the
# content of Figure 3 panel A) as bars with error-bar (T) caps = std over 5 seeds.
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_scaling_bar(algos=None, nrows=2, ncols=3,
                     out_name="bench_scaling_bar.jpg", invert_y=False):
    if algos is None:
        algos = ALGO_ORDER
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_scaling_summary.csv"))

    def extract_n(name):
        return int(name.replace("QM9-gap-", "").replace("k", ""))

    all_ds = sorted(df["dataset"].unique(), key=extract_n)
    ns = [extract_n(d) for d in all_ds]
    FAIL = 25                                       # "failed" row (never reached)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.25 * nrows),
                             sharey=True)
    axflat = np.atleast_1d(axes).flatten()

    for a_idx, algo in enumerate(algos):
        ax = axflat[a_idx]
        means, stds = [], []
        for ds in all_ds:
            sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo)]
            vals = []
            for seed in sub["seed"].unique():
                ssub = sub[sub["seed"] == seed]
                reached = ssub[ssub["pct_opt"] >= 95]
                vals.append(int(reached["round"].min()) + 1
                            if len(reached) > 0 else FAIL)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        color = ALGO_COLORS[algo]
        bars = ax.bar(ns, means, width=3.6, yerr=stds, color=color, alpha=0.7,
                      edgecolor=color, linewidth=0.8,
                      error_kw=dict(ecolor="black", capsize=2.5, elinewidth=1.0,
                                    capthick=1.0))
        # "failed" bars (all 5 seeds never reach 95%) → diagonal-stripe fill
        for rect, m in zip(bars, means):
            if m >= FAIL - 1e-9:
                rect.set_facecolor("none")
                rect.set_hatch("////")
                rect.set_edgecolor(color)
                rect.set_alpha(1.0)
        ax.axhline(FAIL, color="gray", ls=":", lw=0.9, alpha=0.7)
        ax.set_title(_disp(algo), fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel(r"Chemical space size $n$ ($\times 1000$)", fontsize=12)
        if a_idx % ncols == 0:
            ax.set_ylabel(r"$t_{95}$ (rounds)", fontsize=12)
        ax.set_xlim(0, 105)
        # headroom above "failed" (25) so error-bar caps (max mean+std ~29) fit;
        # sharey keeps every subplot on this same range
        ax.set_ylim((FAIL + 6, 0) if invert_y else (0, FAIL + 6))
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 5, 10, 15, 20, FAIL])
        ax.set_yticklabels(["0", "5", "10", "15", "20", "failed"])
        ax.tick_params(labelsize=12)

    # unused cells: bar-type legend in the first one (row 2, col 3 of a
    # 5-algo 2x3 grid); blank the rest
    from matplotlib.patches import Patch
    for j in range(len(algos), len(axflat)):
        axflat[j].axis("off")
    if len(algos) < len(axflat):
        leg_ax = axflat[len(algos)]
        handles = [Patch(facecolor=ALGO_COLORS[a], edgecolor=ALGO_COLORS[a],
                         alpha=0.7, label=_disp(a)) for a in algos]
        handles.append(Patch(facecolor="none", edgecolor="gray", hatch="////",
                             label="failed (never reaches 95%)"))
        leg_ax.legend(handles=handles, loc="center", fontsize=12,
                      frameon=False)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════

def fig_scaling_convergence_grid(out_name="si_fig_scaling_convergence.jpg"):
    """SI figure: per-pool-size convergence curves for all 5 algorithms on the
    QM9-gap scaling study. One subplot per chemical-space size n (5k..100k),
    showing mean +/- std of % of optimum vs round over 5 seeds. 5x4 grid.
    """
    from matplotlib.lines import Line2D
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_scaling_summary.csv"))

    def extract_n(name):
        return int(name.replace("QM9-gap-", "").replace("k", ""))

    all_ds = sorted(df["dataset"].unique(), key=extract_n)
    ns = [extract_n(d) for d in all_ds]

    nrows, ncols = 5, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 18),
                             sharex=True, sharey=True)
    axflat = axes.flatten()

    for idx, (ds, n) in enumerate(zip(all_ds, ns)):
        ax = axflat[idx]
        row, col = divmod(idx, ncols)
        _convergence(ax, df, ds, ALGO_ORDER, ALGO_COLORS, ALGO_LINES,
                     show_legend=False, show_ylabel=(col == 0), fs=11, xmax=20)
        ax.set_title(f"$n = {n}$k", fontsize=13, fontweight="bold")
        if row != nrows - 1:
            ax.set_xlabel("")

    for j in range(len(all_ds), len(axflat)):
        axflat[j].axis("off")

    handles = [Line2D([0], [0], color=ALGO_COLORS[a], ls=ALGO_LINES[a], lw=2.2,
                      label=_disp(a)) for a in ALGO_ORDER]
    handles.append(Line2D([0], [0], color="black", ls="--", lw=1.6, alpha=0.6,
                          label="95% of optimum"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig1_main()
    fig2_scaling()
    fig3_complexity()
    fig4_ratio_scaling()
    fig5_ratio_complexity()
    fig6_trajectory()
    fig3_scaling_boxplot()
    fig3_scaling_boxplot(algos=["GLOSS", "UCB-BO", "BO(EI)", "GA"],
                         nrows=2, ncols=2,
                         out_name="bench_scaling_boxplot_2x2.jpg")
    fig3_scaling_bar()
    fig3_scaling_bar(out_name="bench_scaling_bar_inverted.jpg", invert_y=True)
    fig_scaling_convergence_grid()
    print("\nAll figures generated.")
