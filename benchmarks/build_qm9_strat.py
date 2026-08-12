"""Build quantile-stratified QM9-gap subsets for the new scaling benchmark.

Strategy:
  1. Load the full QM9 CSV, compute 20 RDKit descriptors for every molecule
     with a valid SMILES, drop non-finite rows. This is the "full pool".
  2. Apply min-max normalization to the descriptors using the FULL pool stats
     so every subset shares the same feature scale.
  3. For each target n in [5000, 10000, ..., 100000], split the full pool by
     100 equal-frequency quantile bins on gap, then sample n/100 molecules
     per bin (or take the whole bin when it's smaller). Result: y CDF, y_min,
     y_max, top-95% threshold all stay aligned with the full pool.
  4. Save each subset (X, y, indices into full pool, full pool stats) to
     benchmarks/data/qm9_strat/qm9_strat_n<NNNNN>.npz.

Distinct from the existing load_qm9() path: nothing here writes into
benchmarks/data/qm9.csv or other existing files.
"""

import os
import time

import numpy as np
import pandas as pd

from benchmarks.datasets import _compute_extended_descriptors, _download

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(DATA_DIR, "qm9_strat")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "gap"
SCALES = list(range(5000, 105000, 5000))   # 5k -> 100k step 5k
N_BINS = 100
RNG_SEED = 42
FULL_POOL_NPZ = os.path.join(OUT_DIR, "qm9_full_pool.npz")


def build_full_pool() -> dict:
    """Compute the descriptors-and-target table for the full QM9 CSV once."""
    if os.path.exists(FULL_POOL_NPZ):
        d = np.load(FULL_POOL_NPZ, allow_pickle=False)
        return {k: d[k] for k in d.files}

    print("Loading qm9.csv ...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    csv_path = _download(url, "qm9.csv")
    df = pd.read_csv(csv_path)
    print(f"  raw rows: {len(df)}")

    smiles = df["smiles"].values
    y = df[TARGET].values.astype(float)

    print(f"Computing RDKit descriptors for {len(smiles)} molecules "
          "(this takes a few minutes)...")
    t0 = time.time()
    X_raw, valid = _compute_extended_descriptors(smiles)
    X_raw, y = X_raw[valid], y[valid]
    print(f"  descriptors in {time.time() - t0:.1f}s; valid rows: {len(y)}")

    finite = np.all(np.isfinite(X_raw), axis=1) & np.isfinite(y)
    X_raw, y = X_raw[finite], y[finite]
    print(f"  finite rows: {len(y)}")

    # Global min-max normalization (so every subset shares scale).
    x_min = X_raw.min(axis=0)
    x_max = X_raw.max(axis=0)
    x_rng = np.where(x_max > x_min, x_max - x_min, 1.0)
    X = (X_raw - x_min) / x_rng

    np.savez(
        FULL_POOL_NPZ,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        x_min=x_min, x_max=x_max,
    )
    print(f"Saved: {FULL_POOL_NPZ}")
    return {"X": X.astype(np.float32), "y": y.astype(np.float32),
            "x_min": x_min, "x_max": x_max}


def stratify_sample(y: np.ndarray, n_target: int, n_bins: int,
                    rng: np.random.Generator,
                    thresh_full: float | None = None) -> np.ndarray:
    """Return indices of a quantile-stratified sample of size n_target.

    Always anchors:
      - the global argmin and argmax (so y_min, y_max match the full pool);
      - every molecule >= thresh_full (so the top-95 % bucket is consistent).
    Remaining slots are filled by equal-frequency quantile bins on y.
    """
    N = len(y)
    anchor = {int(np.argmin(y)), int(np.argmax(y))}
    if thresh_full is not None:
        for i in np.where(y >= thresh_full)[0]:
            anchor.add(int(i))
    anchor_idx = np.array(sorted(anchor), dtype=np.int64)
    if len(anchor_idx) >= n_target:
        return np.sort(anchor_idx[:n_target]), 0

    remaining_pool = np.setdiff1d(np.arange(N), anchor_idx, assume_unique=True)
    y_rem = y[remaining_pool]
    need = n_target - len(anchor_idx)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y_rem, quantiles)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bin_assign = np.digitize(y_rem, edges) - 1
    bin_assign = np.clip(bin_assign, 0, n_bins - 1)

    per_bin = need // n_bins
    remainder = need - per_bin * n_bins
    picked = []
    short_bins = 0
    for b in range(n_bins):
        idx_b = np.where(bin_assign == b)[0]
        quota = per_bin + (1 if b >= n_bins - remainder else 0)
        if len(idx_b) <= quota:
            picked.append(idx_b)
            short_bins += int(len(idx_b) < quota)
        else:
            picked.append(rng.choice(idx_b, size=quota, replace=False))
    picked_local = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
    out = np.concatenate([anchor_idx, remaining_pool[picked_local]])
    if len(out) < n_target:
        leftover = np.setdiff1d(np.arange(N), out, assume_unique=False)
        extra = rng.choice(leftover, size=n_target - len(out), replace=False)
        out = np.concatenate([out, extra])
    return np.sort(out), short_bins


def main() -> None:
    pool = build_full_pool()
    X_full = pool["X"]
    y_full = pool["y"]
    N_full = len(y_full)
    print(f"\nFull pool: {N_full} molecules")
    y_min_full = float(y_full.min())
    y_max_full = float(y_full.max())
    thresh_full = y_min_full + 0.95 * (y_max_full - y_min_full)
    n_top_full = int((y_full >= thresh_full).sum())
    print(f"  y range: [{y_min_full:.4f}, {y_max_full:.4f}]")
    print(f"  top-95% threshold: {thresh_full:.4f}")
    print(f"  top-95% count: {n_top_full} ({100*n_top_full/N_full:.4f}%)")

    rng = np.random.default_rng(RNG_SEED)
    summary = []
    for n in SCALES:
        idx, short = stratify_sample(y_full, n, N_BINS, rng,
                                     thresh_full=thresh_full)
        Xs, ys = X_full[idx], y_full[idx]
        # Use the FULL pool threshold for cross-subset consistency.
        n_top = int((ys >= thresh_full).sum())
        out = os.path.join(OUT_DIR, f"qm9_strat_n{n:06d}.npz")
        np.savez(
            out,
            X=Xs, y=ys, indices=idx.astype(np.int32),
            y_min_full=y_min_full, y_max_full=y_max_full,
            thresh_full=thresh_full, n_top_full=n_top_full,
            n_full=N_full,
        )
        print(
            f"  n={n:>6}  short_bins={short}  "
            f"y_range=[{ys.min():.4f}, {ys.max():.4f}]  "
            f"top-95%={n_top} ({100*n_top/n:.4f}%)  "
            f"→ {os.path.basename(out)}"
        )
        summary.append({
            "n": n, "n_top": n_top, "top_pct": 100 * n_top / n,
            "y_min": float(ys.min()), "y_max": float(ys.max()),
            "short_bins": short,
        })

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(OUT_DIR, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
