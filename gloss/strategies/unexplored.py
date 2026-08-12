import warnings
import numpy as np
from gloss.utils import compute_min_distances, is_duplicate


def find_unexplored(surrogate, space, n_points, explored, excluded, direction,
                    tolerance=0.0, unexplored_threshold=None, n_random_samples=10000,
                    distance_metric="euclidean"):
    """Find points in unexplored regions with best predicted values."""
    sign = 1.0 if direction == "maximize" else -1.0

    if space.mode == "discrete":
        candidates = space.get_candidates_excluding(excluded, tolerance)
    else:
        candidates = space.sample(n_random_samples)

    if len(candidates) == 0:
        warnings.warn("unexplored: no candidates available.")
        return []

    if distance_metric == "jaccard":
        from sklearn.neighbors import BallTree
        if len(explored) == 0:
            # No observed points → treat everything as unexplored
            min_dists = np.full(len(candidates), np.inf)
        else:
            _tree = BallTree(explored.astype(float), metric="jaccard")
            min_dists, _ = _tree.query(candidates.astype(float), k=1)
            min_dists = min_dists.ravel()
    else:
        min_dists = compute_min_distances(candidates, explored)

    if unexplored_threshold is None:
        if explored.shape[0] >= 2:
            from scipy.spatial.distance import pdist
            avg_dist = np.mean(pdist(explored))
            unexplored_threshold = avg_dist * 0.5
        else:
            unexplored_threshold = space.diagonal * 0.1

    far_mask = min_dists > unexplored_threshold
    far_candidates = candidates[far_mask]

    if len(far_candidates) == 0:
        warnings.warn(
            f"unexplored: no candidates above threshold {unexplored_threshold:.4f}. "
            "Returning best among all remaining candidates."
        )
        far_candidates = candidates

    preds = surrogate.predict(far_candidates)
    order = np.argsort(sign * preds)[::-1]

    all_excluded = excluded.copy() if excluded.shape[0] > 0 else np.empty((0, space.ndim))
    results = []
    for idx in order:
        if len(results) >= n_points:
            break
        point = far_candidates[idx]
        if is_duplicate(point, all_excluded, tolerance):
            continue
        results.append({
            "point": point.tolist(),
            "strategy": "unexplored",
            "predicted_value": float(preds[idx]),
        })
        all_excluded = np.vstack([all_excluded, point.reshape(1, -1)])

    if len(results) < n_points:
        warnings.warn(f"unexplored: only found {len(results)}/{n_points} points.")
    return results
