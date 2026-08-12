import numpy as np
import pytest
from gloss.utils import (
    compute_min_distances,
    is_duplicate,
    space_diagonal,
)


class TestSpaceDiagonal:
    def test_2d(self):
        bounds = [(0, 3), (0, 4)]
        assert space_diagonal(bounds) == pytest.approx(5.0)

    def test_1d(self):
        bounds = [(0, 10)]
        assert space_diagonal(bounds) == pytest.approx(10.0)


class TestComputeMinDistances:
    def test_basic(self):
        points = np.array([[0, 0], [10, 10]])
        reference = np.array([[0, 0], [5, 5]])
        dists = compute_min_distances(points, reference)
        assert dists[0] == pytest.approx(0.0)
        assert dists[1] == pytest.approx(np.sqrt(50))

    def test_empty_reference(self):
        points = np.array([[0, 0], [1, 1]])
        reference = np.empty((0, 2))
        dists = compute_min_distances(points, reference)
        assert np.all(np.isinf(dists))


class TestIsDuplicate:
    def test_exact_match_discrete(self):
        point = np.array([1, 2, 3])
        excluded = np.array([[1, 2, 3], [4, 5, 6]])
        assert is_duplicate(point, excluded, tolerance=0.0) is True

    def test_no_match_discrete(self):
        point = np.array([1, 2, 4])
        excluded = np.array([[1, 2, 3], [4, 5, 6]])
        assert is_duplicate(point, excluded, tolerance=0.0) is False

    def test_within_tolerance(self):
        point = np.array([1.0001, 2.0, 3.0])
        excluded = np.array([[1.0, 2.0, 3.0]])
        assert is_duplicate(point, excluded, tolerance=0.001) is True

    def test_empty_excluded(self):
        point = np.array([1, 2, 3])
        excluded = np.empty((0, 3))
        assert is_duplicate(point, excluded, tolerance=0.0) is False
