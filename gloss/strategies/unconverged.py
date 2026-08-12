import warnings
import numpy as np
from gloss.utils import compute_min_distances, is_duplicate


def find_unconverged(surrogate, space, n_points, excluded, X_obs,
                     tolerance=0.0, n_random_samples=10000):
    """Find points with highest surrogate uncertainty (pure σ-driven).

    Redesigned from prev-vs-current model comparison to maximum uncertainty
    sampling. Targets regions where the surrogate is least confident,
    complementing global_best (UCB = μ+κσ) which balances both terms.

    Uncertainty source (in priority order):
    1. surrogate.predict(X, return_std=True) → use σ directly
    2. Fallback: min-distance to X_obs (farther = less certain)
    """
    if space.mode == "discrete":
        candidates = space.get_candidates_excluding(excluded, tolerance)
    else:
        candidates = space.sample(n_random_samples)

    if len(candidates) == 0:
        warnings.warn("unconverged: no candidates available.")
        return []

    # Get uncertainty scores
    try:
        mu, sigma = surrogate.predict(candidates, return_std=True)
        uncertainty = sigma
        source = "sigma"
    except TypeError as e:
        if "return_std" not in str(e):
            raise
        # Fallback: distance from observed points as uncertainty proxy
        if X_obs is not None and len(X_obs) > 0:
            uncertainty = compute_min_distances(candidates, np.asarray(X_obs))
        else:
            uncertainty = np.ones(len(candidates))
        mu = surrogate.predict(candidates)
        if X_obs is None or len(X_obs) == 0:
            warnings.warn(
                "unconverged: surrogate does not support return_std and X_obs is empty — "
                "falling back to uniform uncertainty (random selection)."
            )
        source = "distance"

    order = np.argsort(uncertainty)[::-1]

    all_excluded = excluded.copy() if excluded.shape[0] > 0 else np.empty((0, space.ndim))
    results = []
    for idx in order:
        if len(results) >= n_points:
            break
        point = candidates[idx]
        if is_duplicate(point, all_excluded, tolerance):
            continue
        results.append({
            "point": point.tolist(),
            "strategy": "unconverged",
            "predicted_value": float(mu[idx]),
            "uncertainty_score": float(uncertainty[idx]),
            "uncertainty_source": source,
        })
        all_excluded = np.vstack([all_excluded, point.reshape(1, -1)])

    if len(results) < n_points:
        warnings.warn(f"unconverged: only found {len(results)}/{n_points} points.")
    return results
