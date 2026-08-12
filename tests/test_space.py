import numpy as np
import pytest
from gloss.space import SearchSpace


class TestContinuousSpace:
    def test_creation(self):
        space = SearchSpace(
            mode="continuous",
            bounds=[(0, 50), (0, 50), (0, 50)],
        )
        assert space.ndim == 3
        assert space.mode == "continuous"

    def test_diagonal(self):
        space = SearchSpace(mode="continuous", bounds=[(0, 3), (0, 4)])
        assert space.diagonal == pytest.approx(5.0)

    def test_sample_unconstrained(self):
        space = SearchSpace(mode="continuous", bounds=[(0, 10), (0, 10)])
        samples = space.sample(1000)
        assert samples.shape == (1000, 2)
        assert np.all(samples >= 0) and np.all(samples <= 10)

    def test_sample_with_constraints(self):
        space = SearchSpace(
            mode="continuous",
            bounds=[(0, 100), (0, 100), (0, 100)],
            constraints=[{"type": "eq", "fun": lambda x: x[0] + x[1] + x[2] - 100}],
        )
        samples = space.sample(100)
        assert samples.shape[0] > 0
        for s in samples:
            assert abs(s.sum() - 100) < 1e-4

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="min.*max"):
            SearchSpace(mode="continuous", bounds=[(10, 5)])


class TestDiscreteSpaceCandidates:
    def test_from_candidates(self):
        cands = np.array([[1, 2], [3, 4], [5, 6]])
        space = SearchSpace(mode="discrete", candidates=cands)
        assert space.candidates.shape == (3, 2)

    def test_constraint_filtering(self):
        cands = np.array([[10, 90], [50, 50], [90, 10]])
        space = SearchSpace(
            mode="discrete",
            candidates=cands,
            constraints=[{"type": "ineq", "fun": lambda x: x[0] - 20}],
        )
        assert space.candidates.shape[0] == 2


class TestDiscreteSpaceParamGrid:
    def test_from_param_grid(self):
        space = SearchSpace(
            mode="discrete",
            param_grid={"a": [1, 2], "b": [10, 20]},
        )
        assert space.candidates.shape == (4, 2)

    def test_param_grid_with_constraint(self):
        space = SearchSpace(
            mode="discrete",
            param_grid={"a": [1, 2, 3], "b": [1, 2, 3]},
            constraints=[{"type": "eq", "fun": lambda x: x[0] + x[1] - 4}],
        )
        assert space.candidates.shape[0] == 3

    def test_empty_param_grid(self):
        with pytest.raises(ValueError):
            SearchSpace(mode="discrete", param_grid={})


class TestGetExcludedCandidates:
    def test_exclude_from_discrete(self):
        cands = np.array([[1, 2], [3, 4], [5, 6]])
        space = SearchSpace(mode="discrete", candidates=cands)
        remaining = space.get_candidates_excluding(
            excluded=np.array([[3, 4]]), tolerance=0.0
        )
        assert remaining.shape[0] == 2
