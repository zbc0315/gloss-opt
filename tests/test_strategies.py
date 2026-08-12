import numpy as np
import pytest
from gloss.space import SearchSpace
from gloss.strategies.global_best import find_global_best
from gloss.strategies.local_best import find_local_best
from gloss.strategies.unexplored import find_unexplored
from gloss.strategies.unconverged import find_unconverged


class _MockSurrogate:
    """Surrogate that returns negative distance from (5, 5) — minimum at (5, 5)."""
    def predict(self, X):
        return -np.sqrt(((X - 5) ** 2).sum(axis=1))


class TestGlobalBestDiscrete:
    def test_finds_best_point(self):
        cands = np.array([[0, 0], [5, 5], [10, 10], [3, 3]])
        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_global_best(
            surrogate=_MockSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            direction="maximize",
            tolerance=0.0,
        )
        assert len(results) == 1
        np.testing.assert_array_equal(results[0]["point"], [5, 5])

    def test_excludes_existing(self):
        cands = np.array([[0, 0], [5, 5], [10, 10]])
        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_global_best(
            surrogate=_MockSurrogate(),
            space=space,
            n_points=1,
            excluded=np.array([[5, 5]]),
            direction="maximize",
            tolerance=0.0,
        )
        assert len(results) == 1
        assert not np.array_equal(results[0]["point"], [5, 5])

    def test_multiple_points(self):
        cands = np.array([[0, 0], [5, 5], [10, 10], [4, 4]])
        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_global_best(
            surrogate=_MockSurrogate(),
            space=space,
            n_points=2,
            excluded=np.empty((0, 2)),
            direction="maximize",
            tolerance=0.0,
        )
        assert len(results) == 2
        assert not np.array_equal(results[0]["point"], results[1]["point"])


class TestGlobalBestContinuous:
    def test_finds_near_optimum(self):
        space = SearchSpace(mode="continuous", bounds=[(0, 10), (0, 10)])
        results = find_global_best(
            surrogate=_MockSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            direction="maximize",
            n_random_samples=5000,
            n_top=5,
            tolerance=0.01,
        )
        assert len(results) == 1
        point = np.array(results[0]["point"])
        assert np.linalg.norm(point - [5, 5]) < 1.0


class _MultiModalSurrogate:
    """Surrogate with two peaks at (2,2) and (8,8)."""
    def predict(self, X):
        d1 = np.sqrt(((X - 2) ** 2).sum(axis=1))
        d2 = np.sqrt(((X - 8) ** 2).sum(axis=1))
        return np.exp(-d1) + 0.8 * np.exp(-d2)


class TestLocalBestDiscrete:
    def test_finds_local_optima(self):
        grid = np.array([[i, j] for i in range(0, 11) for j in range(0, 11)], dtype=float)
        space = SearchSpace(mode="discrete", candidates=grid)
        results = find_local_best(
            surrogate=_MultiModalSurrogate(),
            space=space,
            n_points=2,
            excluded=np.empty((0, 2)),
            direction="maximize",
            tolerance=0.0,
            window_radius=2.5,
        )
        assert len(results) == 2
        points = np.array([r["point"] for r in results])
        assert any(np.linalg.norm(p - [2, 2]) < 1.5 for p in points)
        assert any(np.linalg.norm(p - [8, 8]) < 1.5 for p in points)

    def test_excludes_points(self):
        grid = np.array([[i, j] for i in range(0, 11) for j in range(0, 11)], dtype=float)
        space = SearchSpace(mode="discrete", candidates=grid)
        results = find_local_best(
            surrogate=_MultiModalSurrogate(),
            space=space,
            n_points=2,
            excluded=np.array([[2, 2]]),
            direction="maximize",
            tolerance=0.0,
            window_radius=2.5,
        )
        for r in results:
            assert not np.array_equal(r["point"], [2, 2])


class TestLocalBestContinuous:
    def test_finds_local_optima(self):
        space = SearchSpace(mode="continuous", bounds=[(0, 10), (0, 10)])
        results = find_local_best(
            surrogate=_MultiModalSurrogate(),
            space=space,
            n_points=2,
            excluded=np.empty((0, 2)),
            direction="maximize",
            tolerance=0.1,
            window_radius=2.0,
            n_random_samples=5000,
        )
        assert len(results) >= 1


class TestUnexploredDiscrete:
    def test_avoids_explored_region(self):
        cands = np.array([[i, 0] for i in range(11)], dtype=float)
        space = SearchSpace(mode="discrete", candidates=cands)
        explored = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        results = find_unexplored(
            surrogate=_MockSurrogate(),
            space=space,
            n_points=1,
            explored=explored,
            excluded=explored,
            direction="maximize",
            tolerance=0.0,
            unexplored_threshold=3.0,
        )
        assert len(results) == 1
        assert results[0]["point"][0] >= 6

    def test_falls_back_when_threshold_too_high(self):
        cands = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        space = SearchSpace(mode="discrete", candidates=cands)
        explored = np.array([[0, 0]], dtype=float)
        with pytest.warns(UserWarning, match="no candidates above threshold"):
            results = find_unexplored(
                surrogate=_MockSurrogate(),
                space=space,
                n_points=1,
                explored=explored,
                excluded=explored,
                direction="maximize",
                tolerance=0.0,
                unexplored_threshold=100.0,
            )
        assert len(results) == 1


class TestUnexploredContinuous:
    def test_explores_away_from_known(self):
        space = SearchSpace(mode="continuous", bounds=[(0, 10), (0, 10)])
        explored = np.array([[1, 1], [2, 2]], dtype=float)
        results = find_unexplored(
            surrogate=_MockSurrogate(),
            space=space,
            n_points=1,
            explored=explored,
            excluded=explored,
            direction="maximize",
            tolerance=0.1,
            n_random_samples=5000,
        )
        assert len(results) == 1
        point = np.array(results[0]["point"])
        assert np.linalg.norm(point - [1, 1]) > 2.0


class _ShiftedSurrogate:
    """Surrogate that returns values shifted by an offset."""
    def __init__(self, offset):
        self.offset = offset
    def predict(self, X):
        return X[:, 0] + self.offset


class TestUnconvergedDiscrete:
    def test_selects_highest_sigma(self):
        """Selects the candidate with highest σ from surrogate."""
        cands = np.array([[1, 0], [5, 0], [10, 0]], dtype=float)
        space = SearchSpace(mode="discrete", candidates=cands)

        class _SigmaSurrogate:
            def predict(self, X, return_std=False):
                mu = X[:, 0].copy()
                if return_std:
                    # sigma = x[0] value, so [10,0] has highest sigma
                    return mu, X[:, 0].copy()
                return mu

        results = find_unconverged(
            surrogate=_SigmaSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            X_obs=np.array([[0.0, 0.0]]),
            tolerance=0.0,
        )
        assert len(results) == 1
        np.testing.assert_array_equal(results[0]["point"], [10, 0])
        assert results[0]["uncertainty_source"] == "sigma"

    def test_fallback_when_no_std(self):
        """Falls back to distance proxy when surrogate has no return_std."""
        cands = np.array([[1, 0], [5, 0], [10, 0]], dtype=float)
        space = SearchSpace(mode="discrete", candidates=cands)
        X_obs = np.array([[0.0, 0.0]])
        results = find_unconverged(
            surrogate=_MockSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            X_obs=X_obs,
            tolerance=0.0,
        )
        assert len(results) == 1
        # [10, 0] is farthest from [0, 0]
        np.testing.assert_array_equal(results[0]["point"], [10, 0])
        assert results[0]["uncertainty_source"] == "distance"

    def test_returns_uncertainty_score(self):
        """Result must contain uncertainty_score key."""
        cands = np.array([[0, 0], [3, 4]], dtype=float)
        space = SearchSpace(mode="discrete", candidates=cands)

        class _StdSurrogate:
            def predict(self, X, return_std=False):
                mu = np.zeros(len(X))
                if return_std:
                    sigma = np.sqrt((X ** 2).sum(axis=1))
                    return mu, sigma
                return mu

        results = find_unconverged(
            surrogate=_StdSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            X_obs=np.array([[0.0, 0.0]]),
            tolerance=0.0,
        )
        assert len(results) == 1
        assert "uncertainty_score" in results[0]
        assert results[0]["uncertainty_score"] == pytest.approx(5.0)
