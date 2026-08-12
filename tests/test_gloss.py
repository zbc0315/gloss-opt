import numpy as np
import pytest
from gloss import GLOSS


class _SimpleSurrogate:
    """Returns sum of features."""
    def predict(self, X):
        return X.sum(axis=1)


class TestGLOSSDiscrete:
    def test_basic_recommend(self):
        cands = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
        gloss = GLOSS(
            space={"candidates": cands},
            mode="discrete",
            direction="maximize",
        )
        results = gloss.recommend(
            surrogate=_SimpleSurrogate(),
            strategy_points={"global_best": 1, "local_best": 1, "unexplored": 1, "unconverged": 1},
        )
        strategies = [r["strategy"] for r in results]
        assert "global_best" in strategies
        assert "local_best" in strategies
        assert "unexplored" in strategies
        unconv = [r for r in results if r["strategy"] == "unconverged"]
        assert len(unconv) == 1
        assert "point" in unconv[0]

    def test_no_duplicate_points(self):
        cands = np.array([[i, 0] for i in range(20)], dtype=float)
        gloss = GLOSS(
            space={"candidates": cands},
            mode="discrete",
            direction="maximize",
        )
        results = gloss.recommend(
            surrogate=_SimpleSurrogate(),
            strategy_points={"global_best": 3, "local_best": 3, "unexplored": 3, "unconverged": 0},
        )
        points = [tuple(r["point"]) for r in results if "point" in r]
        assert len(points) == len(set(points)), "Duplicate points found!"

    def test_excludes_existing(self):
        cands = np.array([[i, 0] for i in range(10)], dtype=float)
        gloss = GLOSS(
            space={"candidates": cands},
            mode="discrete",
            direction="maximize",
        )
        existing = np.array([[9, 0], [8, 0]], dtype=float)
        results = gloss.recommend(
            surrogate=_SimpleSurrogate(),
            X_existing=existing,
            strategy_points={"global_best": 2, "local_best": 0, "unexplored": 0, "unconverged": 0},
        )
        for r in results:
            if "point" in r:
                assert r["point"] != [9, 0] and r["point"] != [8, 0]

    def test_second_call_uses_unconverged(self):
        cands = np.array([[i, 0] for i in range(20)], dtype=float)
        gloss = GLOSS(
            space={"candidates": cands},
            mode="discrete",
            direction="maximize",
        )
        gloss.recommend(
            surrogate=_SimpleSurrogate(),
            strategy_points={"global_best": 1, "local_best": 0, "unexplored": 0, "unconverged": 0},
        )

        class _DifferentSurrogate:
            def predict(self, X):
                return -X.sum(axis=1)

        results = gloss.recommend(
            surrogate=_DifferentSurrogate(),
            strategy_points={"global_best": 0, "local_best": 0, "unexplored": 0, "unconverged": 1},
        )
        unconv = [r for r in results if r["strategy"] == "unconverged"]
        assert len(unconv) == 1
        assert "skipped" not in unconv[0]

    def test_reset_clears_state(self):
        cands = np.array([[i, 0] for i in range(10)], dtype=float)
        gloss = GLOSS(
            space={"candidates": cands},
            mode="discrete",
            direction="maximize",
        )
        gloss.recommend(
            surrogate=_SimpleSurrogate(),
            strategy_points={"global_best": 1, "local_best": 0, "unexplored": 0, "unconverged": 0},
        )
        gloss.reset()
        results = gloss.recommend(
            surrogate=_SimpleSurrogate(),
            strategy_points={"global_best": 0, "local_best": 0, "unexplored": 0, "unconverged": 1},
        )
        unconv = [r for r in results if r["strategy"] == "unconverged"]
        assert len(unconv) == 1
        assert "point" in unconv[0]


class TestGLOSSContinuous:
    def test_basic_recommend(self):
        gloss = GLOSS(
            space={"bounds": [(0, 10), (0, 10)]},
            mode="continuous",
            direction="maximize",
        )
        results = gloss.recommend(
            surrogate=_SimpleSurrogate(),
            strategy_points={"global_best": 1, "local_best": 1, "unexplored": 1, "unconverged": 1},
        )
        strategies = [r["strategy"] for r in results]
        assert "global_best" in strategies


class TestGLOSSValidation:
    def test_no_model_raises(self):
        gloss = GLOSS(
            space={"bounds": [(0, 10)]},
            mode="continuous",
        )
        with pytest.raises(ValueError, match="surrogate.*X_train"):
            gloss.recommend(
                strategy_points={"global_best": 1, "local_best": 0, "unexplored": 0, "unconverged": 0},
            )

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            GLOSS(space={"bounds": [(0, 10)]}, mode="continuous", direction="both")

    def test_mismatched_xy_raises(self):
        gloss = GLOSS(space={"bounds": [(0, 10)]}, mode="continuous")
        with pytest.raises(ValueError, match="same length"):
            gloss.recommend(
                X_train=np.zeros((5, 1)), y_train=np.zeros(3),
                strategy_points={"global_best": 1, "local_best": 0, "unexplored": 0, "unconverged": 0},
            )


class TestGLOSSIntegration:
    def test_full_workflow_discrete(self):
        """Full workflow: auto model selection + all 4 strategies over 2 calls."""
        rng = np.random.default_rng(42)
        cands = np.array([[i, j] for i in range(0, 11, 2) for j in range(0, 11, 2)], dtype=float)
        X_train = rng.choice(len(cands), 15, replace=False)
        X_train = cands[X_train]
        y_train = X_train[:, 0] ** 2 + X_train[:, 1]

        gloss = GLOSS(
            space={"candidates": cands},
            mode="discrete",
            direction="maximize",
        )

        # First call
        results1 = gloss.recommend(
            X_train=X_train, y_train=y_train,
            X_existing=X_train,
            strategy_points={"global_best": 1, "local_best": 1, "unexplored": 1, "unconverged": 1},
        )
        assert len(results1) >= 3  # 3 strategies + 1 skip info

        # Second call with different surrogate to guarantee unconvergence
        class _DiffSurrogate:
            def predict(self, X):
                return -(X[:, 0] ** 2 + X[:, 1])

        results2 = gloss.recommend(
            surrogate=_DiffSurrogate(),
            X_existing=X_train,
            strategy_points={"global_best": 1, "local_best": 1, "unexplored": 1, "unconverged": 1},
        )
        unconv = [r for r in results2 if r["strategy"] == "unconverged" and "skipped" not in r]
        assert len(unconv) >= 1

    def test_full_workflow_continuous_with_constraints(self):
        """Continuous space with sum constraint."""
        rng = np.random.default_rng(42)
        X_train = rng.dirichlet([1, 1, 1], size=20) * 100  # sum = 100
        y_train = X_train[:, 0] * X_train[:, 1]  # maximize product

        gloss = GLOSS(
            space={
                "bounds": [(0, 100), (0, 100), (0, 100)],
                "constraints": [
                    {"type": "eq", "fun": lambda x: x[0] + x[1] + x[2] - 100}
                ],
            },
            mode="continuous",
            direction="maximize",
            n_random_samples=2000,
        )

        results = gloss.recommend(
            surrogate=_SimpleSurrogate(),
            X_existing=X_train,
            strategy_points={"global_best": 2, "local_best": 1, "unexplored": 1, "unconverged": 0},
        )
        assert len(results) >= 1
