# tests/test_strategies_v3.py
import numpy as np
import pytest
from gloss.space import SearchSpace
from gloss.strategies.unconverged import find_unconverged
from gloss.strategies.local_best import find_local_best
from gloss.strategies.global_best import find_global_best


# Shared mock surrogates — defined here so all test classes can use them
class _MockSurrogate:
    """Surrogate: predicted value = negative distance from (5, 5)."""
    def predict(self, X):
        return -np.sqrt(((X - 5) ** 2).sum(axis=1))


class _UncertainSurrogate:
    """Surrogate with high uncertainty far from origin."""
    def predict(self, X, return_std=False):
        mu = -np.sqrt((X ** 2).sum(axis=1))
        if return_std:
            sigma = np.sqrt((X ** 2).sum(axis=1))  # sigma = distance from origin
            return mu, sigma
        return mu


class _NoStdSurrogate:
    """Surrogate that doesn't support return_std."""
    def predict(self, X):
        return -np.sqrt((X ** 2).sum(axis=1))


class TestUnconvergedRedesigned:
    def test_selects_highest_uncertainty_discrete(self):
        """Should select the candidate with highest σ."""
        cands = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 4.0]])  # distances: 0, 1, 5
        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_unconverged(
            surrogate=_UncertainSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            X_obs=np.array([[0.0, 0.0]]),
            tolerance=0.0,
        )
        assert len(results) == 1
        assert results[0]["strategy"] == "unconverged"
        # Point [3,4] has highest σ (distance=5)
        np.testing.assert_allclose(results[0]["point"], [3.0, 4.0])

    def test_no_prev_surrogate_no_longer_needed(self):
        """New signature does NOT take prev_surrogate."""
        import inspect
        sig = inspect.signature(find_unconverged)
        assert "prev_surrogate" not in sig.parameters

    def test_fallback_distance_when_no_std(self):
        """If surrogate lacks return_std, falls back to distance-from-observed proxy."""
        cands = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        space = SearchSpace(mode="discrete", candidates=cands)
        X_obs = np.array([[0.0, 0.0], [1.0, 0.0]])
        results = find_unconverged(
            surrogate=_NoStdSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            X_obs=X_obs,
            tolerance=0.0,
        )
        assert len(results) == 1
        # [10, 0] is farthest from observed → highest uncertainty proxy
        np.testing.assert_allclose(results[0]["point"], [10.0, 0.0])

    def test_excludes_already_excluded(self):
        cands = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_unconverged(
            surrogate=_UncertainSurrogate(),
            space=space,
            n_points=1,
            excluded=np.array([[3.0, 4.0]]),
            X_obs=np.array([[0.0, 0.0]]),
            tolerance=0.0,
        )
        assert len(results) == 1
        assert not np.allclose(results[0]["point"], [3.0, 4.0])

    def test_returns_uncertainty_score_in_result(self):
        cands = np.array([[1.0, 0.0], [3.0, 4.0]])
        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_unconverged(
            surrogate=_UncertainSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            X_obs=np.array([[0.0, 0.0]]),
            tolerance=0.0,
        )
        assert "uncertainty_score" in results[0]
        assert results[0]["uncertainty_score"] > 0

    def test_continuous_space_runs_without_error(self):
        """Continuous mode path should sample and select without crashing."""
        space = SearchSpace(mode="continuous", bounds=[(0.0, 1.0), (0.0, 1.0)])
        results = find_unconverged(
            surrogate=_UncertainSurrogate(),
            space=space,
            n_points=2,
            excluded=np.empty((0, 2)),
            X_obs=np.array([[0.5, 0.5]]),
            tolerance=1e-4,
            n_random_samples=200,
        )
        assert len(results) == 2
        assert all(r["strategy"] == "unconverged" for r in results)

    def test_uncertainty_source_is_sigma_when_std_available(self):
        """When surrogate supports return_std, source should be 'sigma'."""
        cands = np.array([[1.0, 0.0], [3.0, 4.0]])
        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_unconverged(
            surrogate=_UncertainSurrogate(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 2)),
            X_obs=np.array([[0.0, 0.0]]),
            tolerance=0.0,
        )
        assert results[0]["uncertainty_source"] == "sigma"

    def test_empty_x_obs_fallback_warns(self):
        """When X_obs is empty and no return_std, should warn about uniform fallback."""
        cands = np.array([[1.0, 0.0], [3.0, 4.0]])
        space = SearchSpace(mode="discrete", candidates=cands)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            results = find_unconverged(
                surrogate=_NoStdSurrogate(),
                space=space,
                n_points=1,
                excluded=np.empty((0, 2)),
                X_obs=np.empty((0, 2)),
                tolerance=0.0,
            )
        assert len(results) == 1
        assert any("uniform uncertainty" in str(warning.message) for warning in w)


class TestTanimotoDistance:
    """local_best and unexplored should accept distance_metric='jaccard'."""

    def _binary_space(self):
        # 4-bit binary fingerprints
        cands = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ], dtype=float)
        return SearchSpace(mode="discrete", candidates=cands), cands

    def test_local_best_accepts_jaccard_metric(self):
        space, cands = self._binary_space()

        class _PeakAt0:
            def predict(self, X):
                return -np.sum(np.abs(X - np.array([1, 1, 0, 0])), axis=1)

        results = find_local_best(
            surrogate=_PeakAt0(),
            space=space,
            n_points=1,
            excluded=np.empty((0, 4)),
            direction="maximize",
            tolerance=0.0,
            distance_metric="jaccard",
        )
        assert len(results) == 1

    def test_unexplored_accepts_jaccard_metric(self):
        from gloss.strategies.unexplored import find_unexplored

        space, cands = self._binary_space()

        class _Flat:
            def predict(self, X):
                return np.zeros(len(X))

        explored = np.array([[1, 1, 0, 0]], dtype=float)
        results = find_unexplored(
            surrogate=_Flat(),
            space=space,
            n_points=1,
            explored=explored,
            excluded=explored,
            direction="maximize",
            tolerance=0.0,
            distance_metric="jaccard",
        )
        assert len(results) == 1

    def test_euclidean_still_default(self):
        """Calling without distance_metric should work exactly as before."""
        cands = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_local_best(
            surrogate=_MockSurrogate(),
            space=space, n_points=1,
            excluded=np.empty((0, 2)),
            direction="maximize", tolerance=0.0,
        )
        assert len(results) == 1

    def test_unexplored_jaccard_empty_explored_no_crash(self):
        """jaccard path with empty explored should not crash."""
        from gloss.strategies.unexplored import find_unexplored

        space, cands = self._binary_space()

        class _Flat:
            def predict(self, X):
                return np.zeros(len(X))

        results = find_unexplored(
            surrogate=_Flat(),
            space=space,
            n_points=1,
            explored=np.empty((0, 4)),
            excluded=np.empty((0, 4)),
            direction="maximize",
            tolerance=0.0,
            distance_metric="jaccard",
        )
        assert len(results) == 1


class TestBatchDiversity:
    def test_diversity_radius_spreads_batch(self):
        """With large diversity_radius, selected points must be far apart."""
        rng = np.random.default_rng(0)
        group_a = rng.uniform(0, 1, (5, 2)) * 0.1
        group_b = rng.uniform(1, 2, (5, 2)) * 0.1 + 5.0
        cands = np.vstack([group_a, group_b])

        class _FlatSurrogate:
            def predict(self, X, return_std=False):
                mu = np.ones(len(X))
                return (mu, mu) if return_std else mu

        space = SearchSpace(mode="discrete", candidates=cands)
        results = find_global_best(
            surrogate=_FlatSurrogate(),
            space=space, n_points=2,
            excluded=np.empty((0, 2)),
            direction="maximize", tolerance=0.0,
            diversity_radius=2.0,
        )
        assert len(results) == 2
        p0 = np.array(results[0]["point"])
        p1 = np.array(results[1]["point"])
        dist = np.linalg.norm(p0 - p1)
        assert dist >= 2.0, f"Points too close: {dist:.3f}"

    def test_zero_diversity_radius_matches_original(self):
        """diversity_radius=0 should not restrict selection."""
        cands = np.array([[0.0, 0.0], [5.0, 5.0], [5.1, 5.1]])
        space = SearchSpace(mode="discrete", candidates=cands)

        class _MockStd:
            def predict(self, X, return_std=False):
                mu = -np.sqrt(((X - 5) ** 2).sum(axis=1))
                return (mu, np.ones(len(X))) if return_std else mu

        results = find_global_best(
            surrogate=_MockStd(),
            space=space, n_points=2,
            excluded=np.empty((0, 2)),
            direction="maximize", tolerance=0.0,
            diversity_radius=0.0,
        )
        assert len(results) == 2

    def test_diversity_jaccard_metric(self):
        """diversity_metric='jaccard' uses Tanimoto for batch diversity."""
        cands = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
        ], dtype=float)
        space = SearchSpace(mode="discrete", candidates=cands)

        class _FlatStd:
            def predict(self, X, return_std=False):
                mu = np.ones(len(X))
                return (mu, mu) if return_std else mu

        # Jaccard distance between [1,1,0,0] and [1,1,0,1] = 1/4 = 0.25
        # With diversity_radius=0.5 → second selection must differ enough
        results = find_global_best(
            surrogate=_FlatStd(),
            space=space, n_points=2,
            excluded=np.empty((0, 4)),
            direction="maximize", tolerance=0.0,
            diversity_radius=0.5,
            diversity_metric="jaccard",
        )
        # With radius=0.5, [1,1,0,0] and [1,1,0,1] (dist=0.25) can't coexist
        # Should pick [1,1,0,0] and [0,0,1,1] (dist=1.0)
        if len(results) == 2:
            p0 = np.array(results[0]["point"])
            p1 = np.array(results[1]["point"])
            from sklearn.metrics import pairwise_distances
            d = pairwise_distances(p0.reshape(1,-1), p1.reshape(1,-1), metric="jaccard")[0,0]
            assert d >= 0.5

    def test_large_diversity_radius_may_reduce_results(self):
        """If diversity_radius is huge, may return fewer than n_points (with warning)."""
        cands = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        space = SearchSpace(mode="discrete", candidates=cands)

        class _Flat:
            def predict(self, X, return_std=False):
                mu = np.ones(len(X))
                return (mu, mu) if return_std else mu

        import warnings as _w
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            results = find_global_best(
                surrogate=_Flat(),
                space=space, n_points=3,
                excluded=np.empty((0, 2)),
                direction="maximize", tolerance=0.0,
                diversity_radius=10.0,
            )
        assert len(results) <= 3
        if len(results) < 3:
            assert any("diversity_radius" in str(warning.message) for warning in w)
