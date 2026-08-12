import warnings
import numpy as np
from scipy.optimize import minimize
from gloss.utils import is_duplicate


def _ucb_predict(surrogate, X, sign, kappa):
    """Compute UCB/LCB acquisition scores.

    UCB for maximize: mu + kappa*sigma
    LCB for minimize: mu - kappa*sigma
    Unified formula for 'score to maximize descending': sign*mu + kappa*sigma

    Falls back to plain predict if surrogate doesn't support return_std.
    Returns (acquisition_scores, mean_predictions).
    """
    if kappa > 0:
        try:
            mu, sigma = surrogate.predict(X, return_std=True)
            acq = sign * mu + kappa * sigma
            return acq, mu
        except (TypeError, ValueError, AttributeError):
            pass
    preds = surrogate.predict(X)
    return sign * preds, preds


def find_global_best(surrogate, space, n_points, excluded, direction,
                     tolerance=0.0, n_random_samples=10000, n_top=10, kappa=2.0,
                     diversity_radius=0.0, diversity_metric="euclidean"):
    """Find globally optimal points according to surrogate predictions.

    Uses UCB acquisition (mu + kappa*sigma) when the surrogate supports
    uncertainty estimation, otherwise falls back to plain predicted value.
    kappa=0 disables UCB and uses pure predicted mean.

    diversity_radius: minimum distance between selected points in the batch.
        0.0 (default) disables diversity enforcement.
    diversity_metric: 'euclidean' (default) or 'jaccard' for Tanimoto distance.
    """
    sign = 1.0 if direction == "maximize" else -1.0

    if space.mode == "discrete":
        return _discrete_global_best(surrogate, space, n_points, excluded, sign, tolerance, kappa,
                                     diversity_radius, diversity_metric)
    else:
        return _continuous_global_best(
            surrogate, space, n_points, excluded, sign, tolerance,
            n_random_samples, n_top, kappa, diversity_radius, diversity_metric
        )


def _discrete_global_best(surrogate, space, n_points, excluded, sign, tolerance, kappa,
                           diversity_radius=0.0, diversity_metric="euclidean"):
    candidates = space.get_candidates_excluding(excluded, tolerance)
    if len(candidates) == 0:
        warnings.warn("global_best: no candidates available after exclusion.")
        return []

    acq_scores, preds = _ucb_predict(surrogate, candidates, sign, kappa)
    order = np.argsort(acq_scores)[::-1]

    selected_points = []
    results = []

    for idx in order:
        if len(results) >= n_points:
            break
        point = candidates[idx]

        # Diversity check against already-selected points in this batch
        if diversity_radius > 0 and selected_points:
            sel_arr = np.array(selected_points)
            if diversity_metric == "jaccard":
                from sklearn.metrics import pairwise_distances
                dists = pairwise_distances(
                    point.reshape(1, -1).astype(float),
                    sel_arr.astype(float),
                    metric="jaccard"
                ).ravel()
            else:
                dists = np.linalg.norm(sel_arr - point, axis=1)
            if np.any(dists < diversity_radius):
                continue

        results.append({
            "point": point.tolist(),
            "strategy": "global_best",
            "predicted_value": float(preds[idx]),
        })
        selected_points.append(point)

    if len(results) < n_points:
        warnings.warn(f"global_best: only found {len(results)}/{n_points} points "
                      f"(diversity_radius={diversity_radius} may be too large).")
    return results


def _continuous_global_best(surrogate, space, n_points, excluded, sign, tolerance,
                             n_random_samples, n_top, kappa,
                             diversity_radius=0.0, diversity_metric="euclidean"):
    samples = space.sample(n_random_samples)
    if len(samples) == 0:
        warnings.warn("global_best: no feasible samples generated.")
        return []

    acq_scores, preds = _ucb_predict(surrogate, samples, sign, kappa)
    top_indices = np.argsort(acq_scores)[::-1][:n_top]

    scipy_constraints = [
        {"type": c["type"], "fun": c["fun"]}
        for c in space.constraints
    ]
    bounds_scipy = [(b[0], b[1]) for b in space.bounds]

    def objective(x):
        acq, _ = _ucb_predict(surrogate, x.reshape(1, -1), sign, kappa)
        return -float(acq[0])

    refined = []
    for idx in top_indices:
        x0 = samples[idx]
        try:
            res = minimize(
                objective, x0, method="SLSQP",
                bounds=bounds_scipy,
                constraints=scipy_constraints,
                options={"maxiter": 100},
            )
            if res.success:
                point = res.x
                pred_val = surrogate.predict(point.reshape(1, -1))[0]
            else:
                point = x0
                pred_val = preds[idx]
        except Exception:
            point = x0
            pred_val = preds[idx]

        refined.append((point, pred_val))

    refined.sort(key=lambda r: sign * r[1], reverse=True)

    all_excluded = excluded.copy() if excluded.shape[0] > 0 else np.empty((0, space.ndim))
    selected_points = []
    results = []
    for point, pred_val in refined:
        if len(results) >= n_points:
            break
        if is_duplicate(point, all_excluded, tolerance):
            continue

        # Diversity check against already-selected points in this batch
        if diversity_radius > 0 and selected_points:
            sel_arr = np.array(selected_points)
            if diversity_metric == "jaccard":
                from sklearn.metrics import pairwise_distances
                dists = pairwise_distances(
                    point.reshape(1, -1).astype(float),
                    sel_arr.astype(float),
                    metric="jaccard"
                ).ravel()
            else:
                dists = np.linalg.norm(sel_arr - point, axis=1)
            if np.any(dists < diversity_radius):
                continue

        results.append({
            "point": point.tolist(),
            "strategy": "global_best",
            "predicted_value": float(pred_val),
        })
        all_excluded = np.vstack([all_excluded, point.reshape(1, -1)])
        selected_points.append(point)

    if len(results) < n_points:
        remaining_order = np.argsort(sign * preds)[::-1]
        for idx in remaining_order:
            if len(results) >= n_points:
                break
            point = samples[idx]
            if is_duplicate(point, all_excluded, tolerance):
                continue

            # Diversity check in fallback loop
            if diversity_radius > 0 and selected_points:
                sel_arr = np.array(selected_points)
                if diversity_metric == "jaccard":
                    from sklearn.metrics import pairwise_distances
                    dists = pairwise_distances(
                        point.reshape(1, -1).astype(float),
                        sel_arr.astype(float),
                        metric="jaccard"
                    ).ravel()
                else:
                    dists = np.linalg.norm(sel_arr - point, axis=1)
                if np.any(dists < diversity_radius):
                    continue

            results.append({
                "point": point.tolist(),
                "strategy": "global_best",
                "predicted_value": float(preds[idx]),
            })
            all_excluded = np.vstack([all_excluded, point.reshape(1, -1)])
            selected_points.append(point)

    if len(results) < n_points:
        warnings.warn(f"global_best: only found {len(results)}/{n_points} points "
                      f"(diversity_radius={diversity_radius} may be too large).")
    return results
