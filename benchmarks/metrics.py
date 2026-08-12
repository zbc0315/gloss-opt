"""Benchmark metric computation."""

import numpy as np


def best_so_far(y_history):
    """Compute cumulative best observed value.

    Args:
        y_history: list of y values in observation order.

    Returns:
        np.ndarray of cumulative maxima.
    """
    return np.maximum.accumulate(y_history)


def rounds_to_target(best_curve, target):
    """Find the first round where best_curve >= target.

    Args:
        best_curve: array of best-so-far values per round.
        target: target value to reach.

    Returns:
        Round index (0-based) or None if never reached.
    """
    hits = np.where(best_curve >= target)[0]
    if len(hits) > 0:
        return int(hits[0])
    return None


def compute_metrics(round_bests, round_times, y_optimum, n_init_points):
    """Compute all benchmark metrics from a single run.

    Args:
        round_bests: list of best-so-far values at each round end.
        round_times: list of recommendation times per round (seconds).
        y_optimum: true optimum of the dataset/function.
        n_init_points: number of initial points (before rounds start).

    Returns:
        dict with: final_best, target_95_round, avg_time, round_bests, round_times
    """
    target_95 = y_optimum * 0.95 if y_optimum > 0 else y_optimum * 1.05

    return {
        "final_best": round_bests[-1] if len(round_bests) > 0 else None,
        "target_95_round": rounds_to_target(np.array(round_bests), target_95),
        "avg_time": np.mean(round_times) if round_times else 0,
        "round_bests": round_bests,
        "round_times": round_times,
    }
