import warnings
import numpy as np
from itertools import product as cartesian_product
from gloss.utils import space_diagonal, is_duplicate


class SearchSpace:
    """Manages search space definition, constraint filtering, and sampling."""

    def __init__(self, mode, bounds=None, candidates=None, param_grid=None,
                 constraints=None, custom_sampler=None, seed=None):
        if mode not in ("continuous", "discrete"):
            raise ValueError(f"mode must be 'continuous' or 'discrete', got '{mode}'")
        self.mode = mode
        self.constraints = constraints or []
        self.custom_sampler = custom_sampler
        self.rng = np.random.default_rng(seed)

        if mode == "continuous":
            if bounds is None:
                raise ValueError("bounds required for continuous mode")
            bounds = np.array(bounds, dtype=float)
            if np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ValueError("All bounds must have min < max")
            self.bounds = bounds
            self.ndim = len(bounds)
            self.diagonal = space_diagonal(bounds)
            self.candidates = None
        else:
            if candidates is not None:
                self.candidates = np.array(candidates, dtype=float)
            elif param_grid is not None:
                if not param_grid:
                    raise ValueError("param_grid must be non-empty")
                for k, v in param_grid.items():
                    if len(v) == 0:
                        raise ValueError(f"param_grid['{k}'] must be non-empty")
                keys = list(param_grid.keys())
                values = [param_grid[k] for k in keys]
                combos = list(cartesian_product(*values))
                self.candidates = np.array(combos, dtype=float)
            else:
                raise ValueError("discrete mode requires candidates or param_grid")

            if self.constraints:
                self.candidates = self._filter_constraints(self.candidates)

            self.ndim = self.candidates.shape[1]
            self.bounds = np.column_stack([
                self.candidates.min(axis=0),
                self.candidates.max(axis=0),
            ])
            self.diagonal = space_diagonal(self.bounds)

    def _filter_constraints(self, points):
        """Keep only points satisfying all constraints."""
        mask = np.ones(len(points), dtype=bool)
        for c in self.constraints:
            fun = c["fun"]
            ctype = c["type"]
            for i, p in enumerate(points):
                val = fun(p)
                if ctype == "eq":
                    mask[i] &= abs(val) < 1e-6
                elif ctype == "ineq":
                    mask[i] &= val >= -1e-6
        return points[mask]

    def sample(self, n, rng=None):
        """Sample n feasible points from the space."""
        if rng is None:
            rng = self.rng

        if self.custom_sampler is not None:
            return self.custom_sampler(n, self.bounds, self.constraints, rng)

        if self.mode == "discrete":
            indices = rng.choice(len(self.candidates), size=min(n, len(self.candidates)), replace=False)
            return self.candidates[indices]

        if not self.constraints:
            return self._sample_uniform(n, rng)

        samples = self._rejection_sample(n, rng, max_attempts=100 * n)
        if len(samples) >= n:
            return samples[:n]

        eq_constraints = [c for c in self.constraints if c["type"] == "eq"]
        ineq_constraints = [c for c in self.constraints if c["type"] == "ineq"]

        if eq_constraints:
            samples = self._constrained_sample(n, eq_constraints, ineq_constraints, rng)
            if len(samples) >= n:
                return samples[:n]

        if len(samples) > 0:
            warnings.warn(
                f"Could only generate {len(samples)}/{n} feasible samples. "
                "Consider providing a custom_sampler."
            )
            return samples
        raise RuntimeError(
            "Cannot generate feasible samples. Provide a custom_sampler."
        )

    def _sample_uniform(self, n, rng):
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        return rng.uniform(lows, highs, size=(n, self.ndim))

    def _rejection_sample(self, n, rng, max_attempts):
        collected = []
        total_valid = 0
        batch_size = max(n * 10, 1000)
        attempts = 0
        while total_valid < n and attempts < max_attempts:
            batch = self._sample_uniform(batch_size, rng)
            valid = self._filter_constraints(batch)
            if len(valid) > 0:
                collected.append(valid)
                total_valid += len(valid)
            attempts += batch_size
        if collected:
            return np.vstack(collected)
        return np.empty((0, self.ndim))

    def _constrained_sample(self, n, eq_constraints, ineq_constraints, rng):
        collected = []
        for _ in range(n * 100):
            x = self._sample_uniform(1, rng)[0]
            for _ in range(50):
                for c in eq_constraints:
                    val = c["fun"](x)
                    if abs(val) < 1e-6:
                        continue
                    grad = np.zeros(self.ndim)
                    eps = 1e-7
                    for d in range(self.ndim):
                        x_plus = x.copy()
                        x_plus[d] += eps
                        grad[d] = (c["fun"](x_plus) - val) / eps
                    grad_norm = np.dot(grad, grad)
                    if grad_norm > 1e-12:
                        x = x - (val / grad_norm) * grad
                x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
            feasible = True
            for c in eq_constraints:
                if abs(c["fun"](x)) > 1e-4:
                    feasible = False
                    break
            if feasible:
                for c in ineq_constraints:
                    if c["fun"](x) < -1e-6:
                        feasible = False
                        break
            if feasible:
                collected.append(x)
            if len(collected) >= n:
                break
        if collected:
            return np.array(collected)
        return np.empty((0, self.ndim))

    def get_candidates_excluding(self, excluded, tolerance=0.0):
        if self.candidates is None:
            raise ValueError("get_candidates_excluding only for discrete spaces")
        if excluded.shape[0] == 0:
            return self.candidates.copy()
        mask = np.ones(len(self.candidates), dtype=bool)
        for i, c in enumerate(self.candidates):
            if is_duplicate(c, excluded, tolerance):
                mask[i] = False
        return self.candidates[mask]
