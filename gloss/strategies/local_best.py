import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import BallTree
from gloss.utils import is_duplicate


def _make_tree(data, metric="euclidean"):
    """Build a BallTree with the given metric."""
    if metric == "jaccard":
        return BallTree(data.astype(float), metric="jaccard")
    return BallTree(data)


def find_local_best(surrogate, space, n_points, excluded, direction,
                    tolerance=0.0, window_radius=None, n_random_samples=10000,
                    distance_metric="euclidean", top_k=None):
    """Find locally optimal points — better than all neighbors within a window.

    top_k controls the O(K) truncation in discrete mode:
      - None (default): use max(500, n_points * 50) — production default
      - int >= 1: scan exactly top-k candidates by predicted mean
      - 0: scan all candidates (O(n) — no truncation; for ablation)
    Has no effect in continuous mode.
    """
    sign = 1.0 if direction == "maximize" else -1.0

    if window_radius is None:
        window_radius = space.diagonal * 0.1

    if space.mode == "discrete":
        return _discrete_local_best(
            surrogate, space, n_points, excluded, sign, tolerance, window_radius,
            distance_metric, top_k
        )
    else:
        return _continuous_local_best(
            surrogate, space, n_points, excluded, sign, tolerance,
            window_radius, n_random_samples, distance_metric
        )


def _discrete_local_best(surrogate, space, n_points, excluded, sign, tolerance,
                         window_radius, distance_metric="euclidean", top_k=None):
    candidates = space.get_candidates_excluding(excluded, tolerance)
    if len(candidates) == 0:
        warnings.warn("local_best: no candidates available after exclusion.")
        return []

    preds = surrogate.predict(candidates)
    tree = _make_tree(candidates, distance_metric)

    # O(K) truncation: scan only top-k candidates by predicted mean.
    # A local optimum must have a high prediction, so checking only the
    # top candidates is sufficient and avoids O(n) iteration on large pools.
    if top_k is None:
        # production default: same as historical hardcoded behavior
        max_check = min(len(candidates), max(500, n_points * 50))
    elif top_k <= 0:
        # ablation: full O(n) scan
        max_check = len(candidates)
    else:
        # explicit override (for ablation studies)
        max_check = min(len(candidates), int(top_k))
    top_indices = np.argsort(sign * preds)[::-1][:max_check]

    local_optima = []
    for i in top_indices:
        neighbor_indices = tree.query_radius(candidates[i:i+1], r=window_radius)[0]
        neighbors = neighbor_indices[neighbor_indices != i]
        if len(neighbors) == 0:
            local_optima.append((i, preds[i]))
            continue
        is_best = True
        for j in neighbors:
            if sign * preds[j] > sign * preds[i]:
                is_best = False
                break
        if is_best:
            local_optima.append((i, preds[i]))

    local_optima.sort(key=lambda x: sign * x[1], reverse=True)

    results = []
    for idx, pred_val in local_optima:
        if len(results) >= n_points:
            break
        results.append({
            "point": candidates[idx].tolist(),
            "strategy": "local_best",
            "predicted_value": float(pred_val),
        })

    if len(results) < n_points:
        warnings.warn(f"local_best: only found {len(results)}/{n_points} local optima.")
    return results


def _continuous_local_best(surrogate, space, n_points, excluded, sign, tolerance,
                           window_radius, n_random_samples, distance_metric="euclidean"):
    samples = space.sample(n_random_samples)
    if len(samples) == 0:
        warnings.warn("local_best: no feasible samples generated.")
        return []

    preds = surrogate.predict(samples)
    tree = _make_tree(samples, distance_metric)

    candidate_indices = []
    for i in range(len(samples)):
        neighbor_indices = tree.query_radius(samples[i:i+1], r=window_radius)[0]
        neighbors = neighbor_indices[neighbor_indices != i]
        if len(neighbors) == 0:
            candidate_indices.append(i)
            continue
        if np.all(sign * preds[i] >= sign * preds[neighbors]):
            candidate_indices.append(i)

    if not candidate_indices:
        warnings.warn("local_best: no local optima found in random samples.")
        return []

    # In high dimensions, isolated points (no neighbors) are all classified as
    # local optima — cap at top-20 by predicted value to keep scipy calls bounded.
    if len(candidate_indices) > 20:
        candidate_indices = sorted(candidate_indices, key=lambda i: sign * preds[i], reverse=True)[:20]

    scipy_constraints = [
        {"type": c["type"], "fun": c["fun"]}
        for c in space.constraints
    ]
    bounds_scipy = [(b[0], b[1]) for b in space.bounds]

    def objective(x):
        return -sign * surrogate.predict(x.reshape(1, -1))[0]

    refined = []
    for idx in candidate_indices:
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

    merged = []
    merge_dist = window_radius / 2
    for point, pred_val in refined:
        is_dup = False
        for mp, _ in merged:
            if np.linalg.norm(point - mp) < merge_dist:
                is_dup = True
                break
        if not is_dup:
            merged.append((point, pred_val))

    merged.sort(key=lambda r: sign * r[1], reverse=True)

    all_excluded = excluded.copy() if excluded.shape[0] > 0 else np.empty((0, space.ndim))
    results = []
    for point, pred_val in merged:
        if len(results) >= n_points:
            break
        if is_duplicate(point, all_excluded, tolerance):
            continue
        results.append({
            "point": point.tolist(),
            "strategy": "local_best",
            "predicted_value": float(pred_val),
        })
        all_excluded = np.vstack([all_excluded, point.reshape(1, -1)])

    if len(results) < n_points:
        warnings.warn(f"local_best: only found {len(results)}/{n_points} local optima.")
    return results
