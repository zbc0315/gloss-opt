import numpy as np
import pytest
from gloss import GLOSS


class _GPMock:
    """GP mock that supports return_std."""
    def predict(self, X, return_std=False):
        mu = -np.sqrt((X ** 2).sum(axis=1))
        return (mu, np.ones(len(X)) * 0.5) if return_std else mu


def _make_space(n=30):
    cands = np.random.default_rng(0).uniform(0, 10, (n, 2))
    return {"candidates": cands}


class TestAdaptiveKappa:
    def test_fixed_kappa_unchanged(self):
        g = GLOSS(_make_space(), mode="discrete", direction="maximize",
                  ucb_kappa=3.0, kappa_schedule="fixed")
        assert g._get_kappa(0) == 3.0
        assert g._get_kappa(10) == 3.0

    def test_decay_kappa_decreases(self):
        g = GLOSS(_make_space(), mode="discrete", direction="maximize",
                  ucb_kappa=4.0, kappa_min=0.5, kappa_schedule="decay")
        k0 = g._get_kappa(0)
        k5 = g._get_kappa(5)
        k10 = g._get_kappa(10)
        assert k0 > k5 > k10
        assert k10 >= 0.5

    def test_cosine_kappa_shape(self):
        g = GLOSS(_make_space(), mode="discrete", direction="maximize",
                  ucb_kappa=4.0, kappa_min=0.5, kappa_schedule="cosine", n_rounds=20)
        k_start = g._get_kappa(0)
        k_end = g._get_kappa(20)
        assert abs(k_start - 4.0) < 0.01
        assert k_end < 1.0

    def test_round_counter_increments(self):
        cands = np.random.default_rng(0).uniform(0, 10, (50, 2))
        X_train = cands[:10]
        y_train = np.random.default_rng(0).standard_normal(10)
        g = GLOSS({"candidates": cands}, mode="discrete", direction="maximize",
                  ucb_kappa=2.0, kappa_schedule="decay")
        assert g._round == 0
        g.recommend(X_train=X_train, y_train=y_train,
                    strategy_points={"global_best": 1, "local_best": 0,
                                     "unexplored": 0, "unconverged": 0})
        assert g._round == 1

    def test_reset_clears_round_counter(self):
        g = GLOSS(_make_space(), mode="discrete", direction="maximize")
        g._round = 5
        g.reset()
        assert g._round == 0

    def test_invalid_schedule_raises(self):
        g = GLOSS(_make_space(), mode="discrete", direction="maximize",
                  kappa_schedule="invalid")
        with pytest.raises(ValueError, match="kappa_schedule"):
            g._get_kappa(0)


class TestBanditAllocation:
    def _setup(self):
        cands = np.random.default_rng(42).uniform(0, 10, (100, 2))
        g = GLOSS({"candidates": cands}, mode="discrete", direction="maximize",
                  ucb_kappa=2.0, adaptive_strategy=True)
        return g, cands

    def test_feedback_updates_strategy_wins(self):
        g, _ = self._setup()
        results = [
            {"strategy": "global_best",  "point": [1.0, 2.0], "y_actual": 9.5},
            {"strategy": "local_best",   "point": [3.0, 4.0], "y_actual": 5.0},
            {"strategy": "unexplored",   "point": [7.0, 8.0], "y_actual": 3.0},
        ]
        g.feedback(results, current_best_before=4.0)
        assert g._strategy_wins["global_best"] == 1
        assert g._strategy_wins["local_best"]  == 0
        assert g._strategy_wins["unexplored"]  == 0

    def test_bandit_allocation_sums_to_n_points(self):
        g, _ = self._setup()
        for _ in range(5):
            g.feedback([
                {"strategy": "global_best", "point": [1, 1], "y_actual": 9.0},
                {"strategy": "local_best",  "point": [2, 2], "y_actual": 1.0},
            ], current_best_before=0.5)
        allocation = g._bandit_allocation(total_points=g.default_batch_size)
        assert sum(allocation.values()) == g.default_batch_size
        assert allocation["global_best"] >= allocation["local_best"]

    def test_feedback_noop_when_adaptive_false(self):
        cands = np.random.default_rng(0).uniform(0, 10, (30, 2))
        g = GLOSS({"candidates": cands}, mode="discrete",
                  direction="maximize", adaptive_strategy=False)
        g.feedback([{"strategy": "global_best", "point": [1, 1], "y_actual": 5.0}],
                   current_best_before=4.0)
        assert g._strategy_wins is None

    def test_strategy_points_none_uses_bandit(self):
        cands = np.random.default_rng(0).uniform(0, 10, (60, 2))
        g = GLOSS({"candidates": cands}, mode="discrete",
                  direction="maximize", adaptive_strategy=True, ucb_kappa=2.0)
        X_train = cands[:8]
        y_train = np.random.default_rng(0).standard_normal(8)
        recs = g.recommend(X_train=X_train, y_train=y_train)
        assert len(recs) > 0

    def test_reset_clears_bandit_state(self):
        g, _ = self._setup()
        g.feedback([{"strategy": "global_best", "point": [1, 1], "y_actual": 9.0}],
                   current_best_before=0.5)
        g.reset()
        assert g._strategy_wins["global_best"] == 0
        assert g._strategy_tries["global_best"] == 1  # back to Laplace prior

    def test_feedback_minimize_direction(self):
        """feedback() should credit improvement for minimize direction."""
        cands = np.random.default_rng(0).uniform(0, 10, (30, 2))
        g = GLOSS({"candidates": cands}, mode="discrete",
                  direction="minimize", adaptive_strategy=True)
        results = [
            {"strategy": "global_best", "point": [1.0, 2.0], "y_actual": 0.5},
            {"strategy": "local_best",  "point": [3.0, 4.0], "y_actual": 5.0},
        ]
        g.feedback(results, current_best_before=2.0)  # lower is better
        assert g._strategy_wins["global_best"] == 1  # 0.5 < 2.0 → improved
        assert g._strategy_wins["local_best"] == 0


class TestGLOSSV3Integration:
    """End-to-end: all v3 features active together."""

    def test_full_pipeline_discrete(self):
        """GLOSS v3 with Tanimoto + diversity + decay κ + bandit runs without error."""
        rng = np.random.default_rng(99)
        # Binary fingerprint candidates (chemistry-like)
        cands = (rng.uniform(0, 1, (200, 8)) > 0.5).astype(float)
        y_true = -np.sqrt(((cands - 0.5) ** 2).sum(axis=1))

        gloss = GLOSS(
            {"candidates": cands},
            mode="discrete", direction="maximize",
            ucb_kappa=3.0, kappa_schedule="decay", kappa_min=0.5,
            adaptive_strategy=True,
        )

        X_obs = cands[:10]
        y_obs = y_true[:10]

        for _ in range(5):
            recs = gloss.recommend(
                X_train=X_obs, y_train=y_obs,
                strategy_points={
                    "global_best": 4, "local_best": 1,
                    "unexplored": 1, "unconverged": 2,
                },
            )
            points = np.array([r["point"] for r in recs if "point" in r])
            if len(points) == 0:
                break
            new_y = y_true[
                np.array([np.argmin(np.linalg.norm(cands - p, axis=1))
                          for p in points])
            ]
            for r, y in zip(recs, new_y):
                r["y_actual"] = float(y)
            gloss.feedback(recs, current_best_before=float(y_obs.max()))
            X_obs = np.vstack([X_obs, points])
            y_obs = np.append(y_obs, new_y)

        # Should have improved (or at least not crashed)
        assert float(y_obs.max()) >= float(y_true[:10].max())

    def test_continuous_pipeline(self):
        """GLOSS v3 on continuous 6D space with cosine κ."""
        bounds = [(0.0, 1.0)] * 6
        gloss = GLOSS(
            {"bounds": bounds},
            mode="continuous", direction="maximize",
            ucb_kappa=2.0, kappa_schedule="cosine", n_rounds=10,
        )
        rng = np.random.default_rng(0)
        X_obs = rng.uniform(0, 1, (8, 6))
        y_obs = -np.sqrt(((X_obs - 0.5) ** 2).sum(axis=1))

        for _ in range(3):
            recs = gloss.recommend(
                X_train=X_obs, y_train=y_obs,
                strategy_points={"global_best": 4, "local_best": 1,
                                 "unexplored": 1, "unconverged": 2},
            )
            points = np.array([r["point"] for r in recs if "point" in r])
            if len(points):
                new_y = -np.sqrt(((points - 0.5) ** 2).sum(axis=1))
                X_obs = np.vstack([X_obs, points])
                y_obs = np.append(y_obs, new_y)
        assert True  # no error = pass

    def test_gloss_wires_params_to_strategies(self):
        """GLOSS class params distance_metric/diversity_radius reach strategies."""
        cands = np.array([
            [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1],
            [1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0],
            [1, 0, 0, 0], [0, 1, 1, 1],
        ], dtype=float)
        gloss = GLOSS(
            {"candidates": cands},
            mode="discrete", direction="maximize",
            distance_metric="jaccard",
            diversity_radius=0.3,
            diversity_metric="jaccard",
        )
        X_train = cands[:4]
        y_train = np.array([0.5, 0.3, 0.8, 0.2])
        recs = gloss.recommend(
            X_train=X_train, y_train=y_train,
            strategy_points={"global_best": 2, "local_best": 1,
                             "unexplored": 1, "unconverged": 0},
        )
        assert len(recs) >= 1

    def test_reset_fully_clears_state(self):
        """After reset(), round counter and bandit state are cleared."""
        cands = np.random.default_rng(0).uniform(0, 10, (50, 2))
        gloss = GLOSS({"candidates": cands}, mode="discrete",
                      direction="maximize", adaptive_strategy=True,
                      kappa_schedule="decay")
        X = cands[:8]; y = np.ones(8)
        gloss.recommend(X_train=X, y_train=y,
                        strategy_points={"global_best":4,"local_best":1,
                                         "unexplored":1,"unconverged":2})
        assert gloss._round == 1
        gloss.reset()
        assert gloss._round == 0
        assert gloss._strategy_wins["global_best"] == 0
        assert gloss._strategy_tries["global_best"] == 1
