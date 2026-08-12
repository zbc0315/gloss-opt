"""S4 ablation: Local-best top-K truncation on QM9-gap (n=100,000).

For each K in {100, 250, 500, 1000, 2000, 0 (full O(n))}, run GLOSS for
20 rounds × 5 seeds × q=8, measuring per-round wall-clock time and
final pct_opt. Saves bench_local_topk_summary.csv with one row per
(K, seed, round).

Usage:
    python -m benchmarks.bench_local_topk
"""

import os
import time
import numpy as np
import pandas as pd

from benchmarks.datasets import load_qm9
from benchmarks.algorithms import GLOSSAlgorithm
from benchmarks.rf_surrogate import make_rf

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [47, 53, 59, 61, 67]
N_ROUNDS = 20
N_INIT = 8
N_POINTS = 8
GLOSS_STRATEGY = {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0}
GLOSS_DIVERSITY_RADIUS = 0.02
TOP_K_SWEEP = [100, 250, 500, 1000, 2000, 0]   # 0 = full O(n)


def sample_init(y_true, n_init, seed):
    rng = np.random.default_rng(seed)
    threshold = np.quantile(y_true, 0.2)
    pool = np.where(y_true <= threshold)[0]
    return rng.choice(pool, size=n_init, replace=False).tolist()


def run_one(candidates, y_true, top_k, seed, n_rounds=N_ROUNDS):
    """One full GLOSS run; returns list of per-round dicts."""
    y_opt = float(y_true.max())
    y_min = float(y_true.min())

    algo = GLOSSAlgorithm(
        strategy_points=GLOSS_STRATEGY,
        ucb_kappa=2.0,
        diversity_radius=GLOSS_DIVERSITY_RADIUS,
        local_top_k=top_k,
        seed=seed,
    )

    init_idx = sample_init(y_true, N_INIT, seed)
    X_obs = candidates[init_idx].copy()
    y_obs = y_true[init_idx].copy()

    rows = []
    best = float(y_obs.max())
    pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
    rows.append(dict(top_k=top_k, seed=seed, round=-1,
                     best_so_far=best, pct_opt=pct, time_s=0.0))

    for r in range(n_rounds):
        t0 = time.time()
        rf = make_rf(seed=seed * 100 + r)
        rf.fit(X_obs, y_obs)
        indices, _ = algo.recommend_discrete(
            X_obs, y_obs, candidates, y_true, N_POINTS, rf)
        elapsed = time.time() - t0

        if len(indices) == 0:
            rng = np.random.default_rng(seed * 1000 + r)
            avail = list(set(range(len(candidates))) -
                         set(np.argwhere((candidates[:, None] == X_obs).all(-1).any(-1)).flatten().tolist()))
            indices = rng.choice(avail, size=N_POINTS, replace=False).tolist()

        new_X = candidates[indices]
        new_y = y_true[indices]
        X_obs = np.vstack([X_obs, new_X])
        y_obs = np.concatenate([y_obs, new_y])

        best = float(y_obs.max())
        pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
        rows.append(dict(top_k=top_k, seed=seed, round=r,
                         best_so_far=best, pct_opt=pct, time_s=elapsed))
    return rows


def main():
    print("Loading QM9-gap (100k)...")
    ds = load_qm9(n_samples=100000, target="gap")
    candidates, y_true = ds["candidates"], ds["y_true"]
    print(f"  pool: {len(candidates)} candidates, {candidates.shape[1]} features")

    all_rows = []
    for top_k in TOP_K_SWEEP:
        label = "full" if top_k == 0 else f"K={top_k}"
        # K=0 (full O(n)) is ~70× slower per round; run only 1 seed for the
        # timing comparison. Other K values get all 5 seeds for t95 statistics.
        seeds_for_this_k = [47] if top_k == 0 else SEEDS
        for seed in seeds_for_this_k:
            print(f"  [top_k={label}] seed={seed} ...", flush=True)
            rows = run_one(candidates, y_true, top_k, seed)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "bench_local_topk_summary.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")

    # Console summary
    print("\n" + "=" * 70)
    print("Summary (mean over 5 seeds, rounds >= 0)")
    print("=" * 70)
    print(f"{'K':<10} {'mean t/round (s)':<20} {'mean t95 (rounds)':<20} {'reach 95%'}")
    for top_k in TOP_K_SWEEP:
        sub = df[df["top_k"] == top_k]
        per_round = sub[sub["round"] >= 0]
        mean_time = per_round["time_s"].mean()
        seeds_t95 = []
        for seed in SEEDS:
            ssub = per_round[per_round["seed"] == seed]
            r = ssub[ssub["pct_opt"] >= 95]
            seeds_t95.append(int(r["round"].min()) if len(r) > 0 else 20)
        reach = sum(1 for t in seeds_t95 if t < 20)
        label = "full O(n)" if top_k == 0 else str(top_k)
        print(f"{label:<10} {mean_time:<20.3f} {np.mean(seeds_t95):<20.2f} {reach}/5")


if __name__ == "__main__":
    main()
