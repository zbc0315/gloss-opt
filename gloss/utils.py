import numpy as np
from scipy.spatial.distance import cdist


def space_diagonal(bounds):
    """Compute the diagonal length of the bounding box."""
    bounds = np.array(bounds)
    ranges = bounds[:, 1] - bounds[:, 0]
    return np.linalg.norm(ranges)


def compute_min_distances(points, reference):
    """For each point, compute the minimum Euclidean distance to any reference point.

    Returns inf for all points if reference is empty.
    """
    if reference.shape[0] == 0:
        return np.full(points.shape[0], np.inf)
    dists = cdist(points, reference, metric="euclidean")
    return dists.min(axis=1)


def is_duplicate(point, excluded, tolerance=0.0):
    """Check if point is a duplicate of any point in excluded set."""
    if excluded.shape[0] == 0:
        return False
    point = point.reshape(1, -1)
    dists = cdist(point, excluded, metric="euclidean")[0]
    return bool(np.any(dists <= tolerance))
