"""New benchmark suite: 5 studies for GLOSS publication.

Usage:
    python -m benchmarks.bench_main --study main
    python -m benchmarks.bench_main --study scaling
    python -m benchmarks.bench_main --study complexity
    python -m benchmarks.bench_main --study ratio_scaling
    python -m benchmarks.bench_main --study ratio_complexity
    python -m benchmarks.bench_main --study pilot   # quick test (1 seed, 10 rounds)
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from benchmarks.rf_surrogate import make_rf
from benchmarks.algorithms import GLOSSAlgorithm, BOAlgorithm, UCBOAlgorithm, GeneticAlgorithm, RandomAlgorithm
from benchmarks.datasets import load_qm9, load_buchwald_hartwig
from benchmarks.virtual_functions import (
    arrhenius_yield, arrhenius_optimum, COMPLEXITY_LEVELS,
    RATIO_COMPLEXITY_LEVELS, get_virtual_dataset, get_nd_cluster_dataset,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_INIT = 8
N_ROUNDS = 20
N_POINTS = 8
SEEDS = [47, 53, 59, 61, 67]

GLOSS_STRATEGY = {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0}
GLOSS_DIVERSITY_RADIUS = 0.02

GLOSS_RATIOS = {
    "GLOSS(2:3:3)": {"global_best": 2, "local_best": 3, "unexplored": 3, "unconverged": 0},
    "GLOSS(4:2:2)": {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0},
    "GLOSS(6:1:1)": {"global_best": 6, "local_best": 1, "unexplored": 1, "unconverged": 0},
}


def make_algorithms(seed):
    return {
        "GLOSS": GLOSSAlgorithm(
            strategy_points=GLOSS_STRATEGY, ucb_kappa=2.0,
            diversity_radius=GLOSS_DIVERSITY_RADIUS, seed=seed),
        "UCB-BO": UCBOAlgorithm(kappa=2.0, seed=seed),
        "BO(EI)": BOAlgorithm(seed=seed),
        "GA": GeneticAlgorithm(kappa=2.0, seed=seed),
        "Random": RandomAlgorithm(seed=seed),
    }


def make_ratio_algorithms(seed):
    return {
        name: GLOSSAlgorithm(
            strategy_points=sp, ucb_kappa=2.0,
            diversity_radius=GLOSS_DIVERSITY_RADIUS, seed=seed)
        for name, sp in GLOSS_RATIOS.items()
    }


def sample_init(candidates, y_true, n_init, seed):
    """Sample init points from bottom 20%."""
    rng = np.random.default_rng(seed)
    threshold = np.quantile(y_true, 0.2)
    pool = np.where(y_true <= threshold)[0]
    return rng.choice(pool, size=n_init, replace=False).tolist()


def run_discrete(dataset, algo_name, algo, seed, n_rounds=N_ROUNDS):
    """Run one algo on one discrete dataset for one seed.
    Returns (summary_rows, point_rows).
    """
    candidates = dataset["candidates"]
    y_true = dataset["y_true"]
    y_opt = float(y_true.max())
    y_min = float(y_true.min())
    ds_name = dataset["name"]

    init_idx = sample_init(candidates, y_true, N_INIT, seed)
    X_obs = candidates[init_idx].copy()
    y_obs = y_true[init_idx].copy()

    summary = []
    points = []

    best = float(y_obs.max())
    pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
    summary.append({"dataset": ds_name, "algorithm": algo_name, "seed": seed,
                     "round": -1, "best_so_far": best, "pct_opt": pct,
                     "time_s": 0.0, "diversity": 0.0})
    for i, idx in enumerate(init_idx):
        row = {"round": -1, "point_idx": i, "candidate_idx": int(idx),
               "y": float(y_true[idx]), "strategy": "init"}
        # store first 2 features for 2D compatibility
        row["x0"] = float(candidates[idx, 0]) if candidates.shape[1] > 0 else ""
        row["x1"] = float(candidates[idx, 1]) if candidates.shape[1] > 1 else ""
        points.append(row)

    for r in range(n_rounds):
        t0 = time.time()
        rf = make_rf(seed=seed * 100 + r)
        rf.fit(X_obs, y_obs)

        if isinstance(algo, GLOSSAlgorithm):
            indices, strategies = algo.recommend_discrete(
                X_obs, y_obs, candidates, y_true, N_POINTS, rf)
        else:
            indices = algo.recommend_discrete(
                X_obs, y_obs, candidates, y_true, N_POINTS, rf)
            strategies = ["recommended"] * len(indices)

        elapsed = time.time() - t0

        if len(indices) == 0:
            rng = np.random.default_rng(seed * 1000 + r)
            obs_set = set(map(tuple, np.round(X_obs, 10)))
            avail = [i for i in range(len(candidates))
                     if tuple(np.round(candidates[i], 10)) not in obs_set]
            indices = rng.choice(avail, size=min(N_POINTS, len(avail)), replace=False).tolist()
            strategies = ["fallback"] * len(indices)

        new_X = candidates[indices]
        new_y = y_true[indices]
        X_obs = np.vstack([X_obs, new_X])
        y_obs = np.concatenate([y_obs, new_y])

        best = float(y_obs.max())
        pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
        from scipy.spatial.distance import pdist
        div = float(pdist(X_obs[N_INIT:]).mean()) if len(X_obs) > N_INIT + 1 else 0.0

        summary.append({"dataset": ds_name, "algorithm": algo_name, "seed": seed,
                         "round": r, "best_so_far": best, "pct_opt": pct,
                         "time_s": elapsed, "diversity": div})
        for i, (idx, strat) in enumerate(zip(indices, strategies)):
            row = {"round": r, "point_idx": i, "candidate_idx": int(idx),
                   "y": float(y_true[idx]), "strategy": strat}
            row["x0"] = float(candidates[idx, 0]) if candidates.shape[1] > 0 else ""
            row["x1"] = float(candidates[idx, 1]) if candidates.shape[1] > 1 else ""
            points.append(row)

        if r % 5 == 0:
            print(f"    [{algo_name:12s}] seed={seed} r={r:2d}: "
                  f"best={best:.4f} pct={pct:.1f}%  ({elapsed:.1f}s)")

    return summary, points


def run_continuous(func, y_opt, y_min, ds_name, algo_name, algo, seed, n_rounds=N_ROUNDS):
    """Run one algo on continuous [0,1]^2 function."""
    bounds = [(0, 1), (0, 1)]
    rng = np.random.default_rng(seed)

    # Init from bottom 20%
    cand_pool = rng.uniform(0, 1, (50000, 2))
    y_pool = func(cand_pool)
    thresh = np.quantile(y_pool, 0.2)
    low_pool = cand_pool[y_pool <= thresh]
    init_X = low_pool[rng.choice(len(low_pool), size=N_INIT, replace=False)]
    init_y = func(init_X)

    X_obs = init_X.copy()
    y_obs = init_y.copy()

    summary = []
    points = []

    best = float(y_obs.max())
    pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
    summary.append({"dataset": ds_name, "algorithm": algo_name, "seed": seed,
                     "round": -1, "best_so_far": best, "pct_opt": pct,
                     "time_s": 0.0, "diversity": 0.0})
    for i in range(len(init_X)):
        points.append({"round": -1, "point_idx": i, "candidate_idx": "",
                        "y": float(init_y[i]), "strategy": "init",
                        "x0": float(init_X[i, 0]), "x1": float(init_X[i, 1])})

    for r in range(n_rounds):
        t0 = time.time()
        rf = make_rf(seed=seed * 100 + r)
        rf.fit(X_obs, y_obs)

        if isinstance(algo, GLOSSAlgorithm):
            new_X, strategies = algo.recommend_continuous(
                X_obs, y_obs, bounds, N_POINTS, rf)
        elif isinstance(algo, RandomAlgorithm):
            new_X = rng.uniform(0, 1, (N_POINTS, 2))
            strategies = ["random"] * N_POINTS
        else:
            new_X = algo.recommend_continuous(X_obs, y_obs, bounds, N_POINTS, rf)
            strategies = ["recommended"] * N_POINTS

        elapsed = time.time() - t0
        new_y = func(new_X)
        X_obs = np.vstack([X_obs, new_X])
        y_obs = np.concatenate([y_obs, new_y])

        best = float(y_obs.max())
        pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
        from scipy.spatial.distance import pdist
        div = float(pdist(X_obs[N_INIT:]).mean()) if len(X_obs) > N_INIT + 1 else 0.0

        summary.append({"dataset": ds_name, "algorithm": algo_name, "seed": seed,
                         "round": r, "best_so_far": best, "pct_opt": pct,
                         "time_s": elapsed, "diversity": div})
        for i in range(len(new_X)):
            points.append({"round": r, "point_idx": i, "candidate_idx": "",
                            "y": float(new_y[i]), "strategy": strategies[i] if i < len(strategies) else "unknown",
                            "x0": float(new_X[i, 0]), "x1": float(new_X[i, 1])})

        if r % 5 == 0:
            print(f"    [{algo_name:12s}] seed={seed} r={r:2d}: "
                  f"best={best:.4f} pct={pct:.1f}%  ({elapsed:.1f}s)")

    return summary, points


# ═══════════════════════════════════════════════════════════════════════════════
# Study runners
# ═══════════════════════════════════════════════════════════════════════════════

def study_main(seeds=SEEDS, n_rounds=N_ROUNDS):
    """Study 1: 3 datasets × 4 algorithms."""
    print("=" * 60)
    print("STUDY 1: Main Comparison")
    print("=" * 60)

    all_summary = []
    all_points = []

    # --- QM9-gap 100k ---
    print("\n[1/3] Loading QM9-gap (100k)...")
    ds_qm9 = load_qm9(n_samples=100000, target="gap")
    for seed in seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            print(f"  QM9-gap-100k / {aname} / seed={seed}")
            s, p = run_discrete(ds_qm9, aname, algo, seed, n_rounds)
            all_summary.extend(s)
            all_points.extend(p)

    # --- Buchwald-Hartwig ---
    print("\n[2/3] Loading Buchwald-Hartwig...")
    ds_bh = load_buchwald_hartwig()
    for seed in seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            print(f"  Buchwald-Hartwig / {aname} / seed={seed}")
            s, p = run_discrete(ds_bh, aname, algo, seed, n_rounds)
            all_summary.extend(s)
            all_points.extend(p)

    # --- Virtual Arrhenius (discrete mode, 10k candidates) ---
    print("\n[3/3] Virtual Arrhenius (discrete, 10k)...")
    ds_virt = get_virtual_dataset("arrhenius", n_candidates=10000,
                                  n_peaks=3, noise_sigma=0.0, peak_width=0.08, seed=42)
    ds_virt["name"] = "Arrhenius-2D"
    for seed in seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            print(f"  Arrhenius-2D / {aname} / seed={seed}")
            s, p = run_discrete(ds_virt, aname, algo, seed, n_rounds)
            all_summary.extend(s)
            all_points.extend(p)

    df_sum = pd.DataFrame(all_summary)
    df_pts = pd.DataFrame(all_points)
    df_sum.to_csv(os.path.join(RESULTS_DIR, "bench_main_summary.csv"), index=False)
    df_pts.to_csv(os.path.join(RESULTS_DIR, "bench_main_points.csv"), index=False)
    print(f"\nSaved: bench_main_summary.csv, bench_main_points.csv")
    return df_sum


def study_scaling(seeds=SEEDS, n_rounds=N_ROUNDS):
    """Study 2: QM9-gap at n=5k,10k,...,100k × 4 algorithms."""
    print("=" * 60)
    print("STUDY 2: Scaling Study")
    print("=" * 60)

    all_summary = []
    scales = list(range(5000, 105000, 5000))  # 5k to 100k step 5k

    for n in scales:
        print(f"\n--- Scale n={n} ---")
        ds = load_qm9(n_samples=n, target="gap")
        ds["name"] = f"QM9-gap-{n//1000}k"
        for seed in seeds:
            algos = make_algorithms(seed)
            for aname, algo in algos.items():
                print(f"  {ds['name']} / {aname} / seed={seed}")
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)

    df = pd.DataFrame(all_summary)
    df.to_csv(os.path.join(RESULTS_DIR, "bench_scaling_summary.csv"), index=False)
    print(f"\nSaved: bench_scaling_summary.csv")
    return df


def study_complexity(seeds=SEEDS, n_rounds=N_ROUNDS):
    """Study 3: Virtual Arrhenius at 5 complexity levels × 4 algorithms (discrete)."""
    print("=" * 60)
    print("STUDY 3: Complexity Study (discrete, 10k candidates)")
    print("=" * 60)

    all_summary = []

    for level in COMPLEXITY_LEVELS:
        label = level["label"]
        n_peaks = level["n_peaks"]
        noise_sigma = level["noise_sigma"]
        peak_width = level["peak_width"]

        n_cand = level.get("n_candidates", 10000)
        ds = get_virtual_dataset("arrhenius", n_candidates=n_cand,
                                  n_peaks=n_peaks, noise_sigma=noise_sigma,
                                  peak_width=peak_width, seed=42)
        ds["name"] = label
        y = ds["y_true"]
        thresh = y.min() + 0.95 * (ds["y_opt"] - y.min())
        n_top = (y >= thresh).sum()
        p8 = (1 - (1 - n_top / len(y)) ** 8) * 100

        print(f"\n--- {label} (peaks={n_peaks}, noise={noise_sigma}, width={peak_width}, n={n_cand}) ---")
        print(f"    y_opt={ds['y_opt']:.2f}, n_top={n_top}, P(8)={p8:.2f}%")

        for seed in seeds:
            algos = make_algorithms(seed)
            for aname, algo in algos.items():
                print(f"  {label} / {aname} / seed={seed}")
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)

    df = pd.DataFrame(all_summary)
    df.to_csv(os.path.join(RESULTS_DIR, "bench_complexity_summary.csv"), index=False)
    print(f"\nSaved: bench_complexity_summary.csv")
    return df


def study_ratio_scaling(seeds=SEEDS, n_rounds=N_ROUNDS):
    """Study 4: QM9-gap at n=5k,...,100k × 3 GLOSS ratios."""
    print("=" * 60)
    print("STUDY 4: Ratio Scaling Study")
    print("=" * 60)

    all_summary = []
    scales = list(range(5000, 105000, 5000))

    for n in scales:
        print(f"\n--- Scale n={n} ---")
        ds = load_qm9(n_samples=n, target="gap")
        ds["name"] = f"QM9-gap-{n//1000}k"
        for seed in seeds:
            algos = make_ratio_algorithms(seed)
            for aname, algo in algos.items():
                print(f"  {ds['name']} / {aname} / seed={seed}")
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)

    df = pd.DataFrame(all_summary)
    df.to_csv(os.path.join(RESULTS_DIR, "bench_ratio_scaling_summary.csv"), index=False)
    print(f"\nSaved: bench_ratio_scaling_summary.csv")
    return df


def study_ratio_complexity(seeds=SEEDS, n_rounds=N_ROUNDS):
    """Study 5: nD clusters with increasing dimensionality × 3 GLOSS ratios."""
    print("=" * 60)
    print("STUDY 5: Ratio Complexity (nD clusters, dim increases)")
    print("=" * 60)

    all_summary = []

    for level in RATIO_COMPLEXITY_LEVELS:
        label = level["label"]
        ndim = level["ndim"]
        n_peaks = level["n_peaks"]
        peak_width = level["peak_width"]
        n_cand = level["n_candidates"]

        ds = get_nd_cluster_dataset(ndim=ndim, n_peaks=n_peaks,
                                     peak_width=peak_width,
                                     n_candidates=n_cand, seed=42)
        ds["name"] = label
        y = ds["y_true"]
        thresh = y.min() + 0.95 * (ds["y_opt"] - y.min())
        n_top = (y >= thresh).sum()
        p8 = (1 - (1 - n_top / len(y)) ** 8) * 100

        print(f"\n--- {label} (ndim={ndim}, peaks={n_peaks}, n={n_cand}) ---")
        print(f"    n_top={n_top}, P(8)={p8:.2f}%")

        for seed in seeds:
            algos = make_ratio_algorithms(seed)
            for aname, algo in algos.items():
                print(f"  {label} / {aname} / seed={seed}")
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)

    df = pd.DataFrame(all_summary)
    df.to_csv(os.path.join(RESULTS_DIR, "bench_ratio_complexity_summary.csv"), index=False)
    print(f"\nSaved: bench_ratio_complexity_summary.csv")
    return df


def study_pilot():
    """Quick pilot: 1 seed, 10 rounds, all 3 datasets."""
    print("=" * 60)
    print("PILOT: Quick validation (1 seed, 10 rounds)")
    print("=" * 60)

    pilot_seeds = [47]
    pilot_rounds = 10
    all_summary = []

    # QM9-gap 20k (quick)
    print("\n[1/3] QM9-gap 20k...")
    ds = load_qm9(n_samples=20000, target="gap")
    ds["name"] = "QM9-gap-20k"
    for seed in pilot_seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            s, _ = run_discrete(ds, aname, algo, seed, pilot_rounds)
            all_summary.extend(s)

    # Buchwald-Hartwig
    print("\n[2/3] Buchwald-Hartwig...")
    ds_bh = load_buchwald_hartwig()
    for seed in pilot_seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            s, _ = run_discrete(ds_bh, aname, algo, seed, pilot_rounds)
            all_summary.extend(s)

    # Virtual Arrhenius (discrete)
    print("\n[3/3] Arrhenius-2D (discrete, 10k)...")
    ds_virt = get_virtual_dataset("arrhenius", n_candidates=10000,
                                  n_peaks=3, noise_sigma=0.0, peak_width=0.08, seed=42)
    ds_virt["name"] = "Arrhenius-2D"
    for seed in pilot_seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            s, _ = run_discrete(ds_virt, aname, algo, seed, pilot_rounds)
            all_summary.extend(s)

    # Print results
    df = pd.DataFrame(all_summary)
    print("\n" + "=" * 60)
    print("PILOT RESULTS")
    print("=" * 60)
    for ds_name in df["dataset"].unique():
        print(f"\n--- {ds_name} ---")
        sub = df[df["dataset"] == ds_name]
        final = sub[sub["round"] == sub["round"].max()]
        for _, row in final.sort_values("pct_opt", ascending=False).iterrows():
            print(f"  {row['algorithm']:12s}: pct_opt={row['pct_opt']:.1f}%  best={row['best_so_far']:.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="pilot",
                        choices=["pilot", "main", "scaling", "complexity",
                                 "ratio_scaling", "ratio_complexity", "all"])
    args = parser.parse_args()

    if args.study == "pilot":
        study_pilot()
    elif args.study == "main":
        study_main()
    elif args.study == "scaling":
        study_scaling()
    elif args.study == "complexity":
        study_complexity()
    elif args.study == "ratio_scaling":
        study_ratio_scaling()
    elif args.study == "ratio_complexity":
        study_ratio_complexity()
    elif args.study == "all":
        study_main()
        study_scaling()
        study_complexity()
        study_ratio_scaling()
        study_ratio_complexity()
