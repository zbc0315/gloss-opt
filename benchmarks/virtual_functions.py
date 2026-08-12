"""Chemistry-inspired 2D virtual functions for GLOSS benchmarking.

Three function families with tunable complexity:
  1. Arrhenius Yield Surface — reaction yield f(temperature, catalyst_loading)
  2. Muller-Brown Potential — reaction energy landscape
  3. Lennard-Jones Surface — intermolecular interaction landscape

All functions:
  - Domain: [0, 1] x [0, 1]
  - Output: positive float (suitable for maximization)
  - Tunable: n_peaks, noise_sigma, peak_width
  - Accept numpy arrays of shape (n, 2), return shape (n,)
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Arrhenius Yield Surface
# ═══════════════════════════════════════════════════════════════════════════════

def arrhenius_yield(X, n_peaks=3, noise_sigma=0.0, peak_width=0.08, seed=42):
    """Reaction yield surface: y(T, C) with Arrhenius kinetics + degradation.

    x1 -> temperature (mapped to 300-700K)
    x2 -> catalyst loading (0-1)

    Each "peak" represents a different catalytic regime with its own
    activation energy, optimal temperature, and selectivity window.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)
    T = 300 + 400 * X[:, 0]   # temperature in K
    C = X[:, 1]                # catalyst loading 0-1

    # Generate peak parameters deterministically from seed
    rng_params = np.random.RandomState(seed + 100)
    Ea_values = rng_params.uniform(40, 80, n_peaks)        # activation energy (kJ/mol)
    T_opt = rng_params.uniform(400, 650, n_peaks)           # optimal temperature
    C_opt = rng_params.uniform(0.2, 0.8, n_peaks)           # optimal catalyst loading
    # Global peak is clearly dominant; secondary peaks decay as 1/(i+1)
    heights = np.array([10.0 / (1 + 0.8 * i) for i in range(n_peaks)])
    # peak 0 = 10.0, peak 1 = 5.56, peak 2 = 3.85, ... → 95% threshold (~9.5) only reachable at peak 0

    R = 8.314e-3  # gas constant kJ/(mol·K)
    sigma_T = peak_width * 400  # peak width in K units

    y = np.zeros(len(X))
    for i in range(n_peaks):
        # Arrhenius rise
        arrh = np.exp(-Ea_values[i] / (R * T))
        arrh_opt = np.exp(-Ea_values[i] / (R * T_opt[i]))
        arrh_norm = arrh / (arrh_opt + 1e-30)  # normalize so peak=1 at T_opt

        # Thermal degradation envelope (Gaussian around T_opt)
        degrad = np.exp(-0.5 * ((T - T_opt[i]) / sigma_T) ** 2)

        # Catalyst loading effect (Gaussian around C_opt)
        sigma_C = peak_width
        cat_eff = np.exp(-0.5 * ((C - C_opt[i]) / sigma_C) ** 2)

        y += heights[i] * arrh_norm * degrad * cat_eff

    # Weak background (baseline reactivity)
    y += 0.5 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])

    # Noise
    if noise_sigma > 0:
        y += rng.normal(0, noise_sigma, len(y))

    return np.maximum(y, 0)


def arrhenius_optimum(n_peaks=3, peak_width=0.08, seed=42):
    """Find the optimum of the Arrhenius yield surface by grid search."""
    res = 500
    xs = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    vals = arrhenius_yield(pts, n_peaks=n_peaks, noise_sigma=0.0,
                           peak_width=peak_width, seed=seed)
    idx = np.argmax(vals)
    return float(vals[idx]), pts[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Muller-Brown Potential (rescaled)
# ═══════════════════════════════════════════════════════════════════════════════

def muller_brown(X, n_peaks=3, noise_sigma=0.0, peak_width=0.08, seed=42):
    """Multi-well potential on [0,1]^2, inspired by Muller-Brown.

    Uses Gaussian wells with tunable positions, depths, and widths.
    n_peaks: number of wells (global optimum is the deepest).
    peak_width: Gaussian sigma for each well.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)

    rng_params = np.random.RandomState(seed + 200)
    centers = rng_params.uniform(0.12, 0.88, (n_peaks, 2))
    heights = np.linspace(10, 5, n_peaks)
    heights[0] = 10.0  # global optimum

    y = np.zeros(len(X))
    for i in range(n_peaks):
        dx = X[:, 0] - centers[i, 0]
        dy = X[:, 1] - centers[i, 1]
        # Anisotropic Gaussian (chemistry: different sensitivity per axis)
        angle = rng_params.uniform(0, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        sigma_u = peak_width * rng_params.uniform(0.8, 1.2)
        sigma_v = peak_width * rng_params.uniform(0.5, 1.0)
        y += heights[i] * np.exp(-0.5 * (u / sigma_u) ** 2
                                  - 0.5 * (v / sigma_v) ** 2)

    # Non-trivial background (quadratic bowl + ripple)
    y += 1.0 - 0.5 * ((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2)
    y += 0.3 * np.sin(3 * np.pi * X[:, 0]) * np.sin(3 * np.pi * X[:, 1])

    if noise_sigma > 0:
        y += rng.normal(0, noise_sigma, len(y))

    return np.maximum(y, 0)


def muller_brown_optimum(n_peaks=3, peak_width=1.0, seed=42):
    """Find the optimum of the Muller-Brown surface by grid search."""
    res = 500
    xs = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    vals = muller_brown(pts, n_peaks=n_peaks, noise_sigma=0.0,
                        peak_width=peak_width, seed=seed)
    idx = np.argmax(vals)
    return float(vals[idx]), pts[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Lennard-Jones Interaction Surface
# ═══════════════════════════════════════════════════════════════════════════════

def lennard_jones(X, n_peaks=3, noise_sigma=0.0, peak_width=0.08, seed=42):
    """Morse-potential inspired surface with multiple interaction wells.

    Models intermolecular interactions using Morse potential form (numerically
    stable alternative to LJ 12-6). Each well has a steep repulsive side and
    a gentle attractive tail — asymmetric peaks typical of chemistry.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)

    rng_params = np.random.RandomState(seed + 300)
    centers = rng_params.uniform(0.15, 0.85, (n_peaks, 2))
    depths = np.linspace(10, 5, n_peaks)
    depths[0] = 10.0

    y = np.zeros(len(X))
    for i in range(n_peaks):
        r = np.sqrt(np.sum((X - centers[i]) ** 2, axis=1))
        # Morse potential: D * (1 - exp(-a*(r-re)))^2, negated for maximization
        re = peak_width * 0.5  # equilibrium distance
        a = 1.0 / peak_width   # controls well width
        morse = depths[i] * (1 - np.exp(-a * (r - re))) ** 2
        y += depths[i] - morse  # peak at r=re, decays to 0 far away

    # Smooth background
    y += 1.0 + 0.3 * np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])

    if noise_sigma > 0:
        y += rng.normal(0, noise_sigma, len(y))

    return np.maximum(y, 0)


def lennard_jones_optimum(n_peaks=3, peak_width=0.08, seed=42):
    """Find the optimum of the LJ surface by grid search."""
    res = 500
    xs = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    vals = lennard_jones(pts, n_peaks=n_peaks, noise_sigma=0.0,
                         peak_width=peak_width, seed=seed)
    idx = np.argmax(vals)
    return float(vals[idx]), pts[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Complexity configurations for benchmark
# ═══════════════════════════════════════════════════════════════════════════════

# Default virtual function for Study 1 (main comparison)
DEFAULT_FUNC = "arrhenius"
DEFAULT_PARAMS = {"n_peaks": 3, "noise_sigma": 0.0, "peak_width": 0.08}

# Complexity levels for Study 3 and 5
COMPLEXITY_LEVELS = [
    {"label": "C1-simple",    "n_peaks": 1, "noise_sigma": 0.0,  "peak_width": 0.15, "n_candidates": 5000},
    {"label": "C2-moderate",  "n_peaks": 3, "noise_sigma": 0.0,  "peak_width": 0.10, "n_candidates": 10000},
    {"label": "C3-multi",     "n_peaks": 5, "noise_sigma": 0.05, "peak_width": 0.07, "n_candidates": 20000},
    {"label": "C4-complex",   "n_peaks": 8, "noise_sigma": 0.1,  "peak_width": 0.05, "n_candidates": 30000},
    {"label": "C5-extreme",   "n_peaks": 12, "noise_sigma": 0.15, "peak_width": 0.03, "n_candidates": 50000},
]

def arrhenius_multicluster(X, n_peaks=3, noise_sigma=0.0, peak_width=0.08, seed=42):
    """Multi-cluster variant: all peaks have EQUAL height (~10.0).

    For ratio study: discovering ANY cluster gives near-optimal value.
    More exploration → more cluster discovery → better performance for 2:3:3.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)
    T = 300 + 400 * X[:, 0]
    C = X[:, 1]

    rng_params = np.random.RandomState(seed + 100)
    Ea_values = rng_params.uniform(40, 80, n_peaks)
    T_opt = rng_params.uniform(400, 650, n_peaks)
    C_opt = rng_params.uniform(0.2, 0.8, n_peaks)

    # Equal heights: all peaks at 10.0 ± small variation
    heights = np.full(n_peaks, 10.0) + rng_params.uniform(-0.3, 0.3, n_peaks)

    R = 8.314e-3
    sigma_T = peak_width * 400

    y = np.zeros(len(X))
    for i in range(n_peaks):
        arrh = np.exp(-Ea_values[i] / (R * T))
        arrh_opt = np.exp(-Ea_values[i] / (R * T_opt[i]))
        arrh_norm = arrh / (arrh_opt + 1e-30)
        degrad = np.exp(-0.5 * ((T - T_opt[i]) / sigma_T) ** 2)
        sigma_C = peak_width
        cat_eff = np.exp(-0.5 * ((C - C_opt[i]) / sigma_C) ** 2)
        y += heights[i] * arrh_norm * degrad * cat_eff

    y += 0.5 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])

    if noise_sigma > 0:
        y += rng.normal(0, noise_sigma, len(y))

    return np.maximum(y, 0)


def arrhenius_multicluster_optimum(n_peaks=3, peak_width=0.08, seed=42):
    res = 500
    xs = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    vals = arrhenius_multicluster(pts, n_peaks=n_peaks, noise_sigma=0.0,
                                  peak_width=peak_width, seed=seed)
    idx = np.argmax(vals)
    return float(vals[idx]), pts[idx]


def gaussian_clusters(X, n_peaks=3, noise_sigma=0.0, peak_width=0.08, seed=42):
    """Equal-height Gaussian clusters for ratio study.

    Simple: y = max_i(h * exp(-||x - c_i||^2 / (2*sigma^2))) + background
    All peaks have height 10.0. Finding ANY peak gives ~95%.
    More exploration = more cluster discovery = faster convergence.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)

    rng_params = np.random.RandomState(seed + 500)
    centers = rng_params.uniform(0.1, 0.9, (n_peaks, 2))

    y = np.zeros(len(X))
    for i in range(n_peaks):
        d2 = np.sum((X - centers[i]) ** 2, axis=1)
        y = np.maximum(y, 10.0 * np.exp(-d2 / (2 * peak_width ** 2)))

    # Weak background
    y += 0.5 + 0.2 * np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])

    if noise_sigma > 0:
        y += rng.normal(0, noise_sigma, len(y))

    return np.maximum(y, 0)


def gaussian_clusters_optimum(n_peaks=3, peak_width=0.08, seed=42):
    res = 500
    xs = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    vals = gaussian_clusters(pts, n_peaks=n_peaks, noise_sigma=0.0,
                             peak_width=peak_width, seed=seed)
    idx = np.argmax(vals)
    return float(vals[idx]), pts[idx]


# Ratio study: equal-height clusters in increasing dimensions
# Higher dimensionality = curse of dimensionality = exploration becomes essential
RATIO_COMPLEXITY_LEVELS = [
    {"label": "RC1-2D-3C",  "ndim": 2,  "n_peaks": 3, "peak_width": 0.10, "n_candidates": 5000},
    {"label": "RC2-4D-5C",  "ndim": 4,  "n_peaks": 5, "peak_width": 0.08, "n_candidates": 10000},
    {"label": "RC3-6D-5C",  "ndim": 6,  "n_peaks": 5, "peak_width": 0.15, "n_candidates": 20000},
    {"label": "RC4-8D-7C",  "ndim": 8,  "n_peaks": 7, "peak_width": 0.14, "n_candidates": 30000},
    {"label": "RC5-10D-7C", "ndim": 10, "n_peaks": 7, "peak_width": 0.20, "n_candidates": 50000},
]


def nd_clusters(X, n_peaks=5, ndim=4, peak_width=0.12, noise_sigma=0.0, seed=42):
    """Equal-height Gaussian clusters in nD space.

    As dimensionality increases, random coverage drops exponentially
    (curse of dimensionality), making active exploration essential.
    All peaks have height ~10.0 so finding ANY cluster gives ~95%.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)
    assert X.shape[1] == ndim, f"Expected {ndim}D input, got {X.shape[1]}D"

    rng_params = np.random.RandomState(seed + 600)
    centers = rng_params.uniform(0.15, 0.85, (n_peaks, ndim))

    # Use key-dimension distance (subset of dims) to avoid curse of dimensionality
    # in the distance computation itself
    n_key = max(2, ndim // 2)
    key_dims = rng_params.choice(ndim, size=n_key, replace=False)

    y = np.zeros(len(X))
    for i in range(n_peaks):
        # Distance only in key dimensions
        d2 = np.sum((X[:, key_dims] - centers[i, key_dims]) ** 2, axis=1)
        y = np.maximum(y, 10.0 * np.exp(-d2 / (2 * peak_width ** 2)))

    # Weak background
    y += 0.5

    if noise_sigma > 0:
        y += rng.normal(0, noise_sigma, len(y))

    return np.maximum(y, 0)


def nd_clusters_optimum(n_peaks=5, ndim=4, peak_width=0.12, seed=42):
    """Optimum is always 10.5 (peak height 10.0 + background 0.5)."""
    return 10.5, None


def get_nd_cluster_dataset(ndim=4, n_peaks=5, peak_width=0.12,
                           n_candidates=10000, seed=42):
    """Generate nD cluster dataset for ratio study."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_candidates, ndim))
    y = nd_clusters(X, n_peaks=n_peaks, ndim=ndim,
                    peak_width=peak_width, seed=seed)
    y_opt = 10.5  # peak (10.0) + background (0.5)
    return {
        "candidates": X,
        "y_true": y,
        "y_opt": y_opt,
        "name": f"nD-clusters-{ndim}D-{n_peaks}C",
    }


FUNC_REGISTRY = {
    "arrhenius": (arrhenius_yield, arrhenius_optimum),
    "arrhenius_mc": (arrhenius_multicluster, arrhenius_multicluster_optimum),
    "gaussian_clusters": (gaussian_clusters, gaussian_clusters_optimum),
    "muller_brown": (muller_brown, muller_brown_optimum),
    "lennard_jones": (lennard_jones, lennard_jones_optimum),
}


def get_virtual_dataset(func_name="arrhenius", n_candidates=10000,
                        n_peaks=3, noise_sigma=0.0, peak_width=0.08, seed=42):
    """Generate a discrete virtual dataset for benchmarking.

    Returns dict with keys: candidates (n, 2), y_true (n,), y_opt, x_opt
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_candidates, 2))

    func, opt_func = FUNC_REGISTRY[func_name]
    y = func(X, n_peaks=n_peaks, noise_sigma=noise_sigma,
             peak_width=peak_width, seed=seed)
    y_opt, x_opt = opt_func(n_peaks=n_peaks, peak_width=peak_width, seed=seed)

    return {
        "candidates": X,
        "y_true": y,
        "y_opt": y_opt,
        "x_opt": x_opt,
        "name": f"{func_name}_p{n_peaks}_w{peak_width}_n{noise_sigma}",
    }


if __name__ == "__main__":
    # Quick validation
    for fname in ["arrhenius", "muller_brown", "lennard_jones"]:
        for level in COMPLEXITY_LEVELS:
            ds = get_virtual_dataset(fname, n_candidates=5000, **{k: v for k, v in level.items() if k != "label"})
            y = ds["y_true"]
            y_opt = ds["y_opt"]
            thresh = y.min() + 0.95 * (y_opt - y.min())
            n_top = (y >= thresh).sum()
            p8 = (1 - (1 - n_top / len(y)) ** 8) * 100
            print(f"{fname:15s} {level['label']:12s}  y=[{y.min():.2f}, {y.max():.2f}]  "
                  f"y_opt={y_opt:.2f}  n_top={n_top:3d}  P(8)={p8:.2f}%")
