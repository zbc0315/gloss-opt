"""Unified algorithm interfaces for benchmarking.

All algorithms use the same fixed-hyperparameter GP surrogate:
  ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0), alpha=1e-5

Each algorithm exposes:
  recommend(X_observed, y_observed, candidates_or_space, n_points) -> indices or points
"""

import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm


def make_fixed_gp():
    """Create a GP with fixed hyperparameters (no kernel optimization)."""
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        nu=2.5, length_scale=1.0, length_scale_bounds="fixed"
    )
    return GaussianProcessRegressor(
        kernel=kernel, alpha=1e-5, optimizer=None, normalize_y=True
    )


def fit_gp(X, y):
    """Fit a fixed-hyperparameter GP on observed data."""
    gp = make_fixed_gp()
    gp.fit(X, y)
    return gp


# --------------------------------------------------------------------------
# GLOSS
# --------------------------------------------------------------------------
class GLOSSAlgorithm:
    """GLOSS wrapper for benchmarking."""

    def __init__(self, strategy_points=None, ucb_kappa=2.0, diversity_radius=0.0,
                 diversity_metric="euclidean", local_top_k=None, seed=None):
        from gloss import GLOSS
        self._gloss_cls = GLOSS
        self._strategy_points = strategy_points
        self._ucb_kappa = ucb_kappa
        self._diversity_radius = diversity_radius
        self._diversity_metric = diversity_metric
        self._local_top_k = local_top_k
        self._seed = seed
        self._call_count = 0

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Recommend from discrete candidates.

        Returns:
            (all_indices, global_indices): all recommended indices and global_best-only indices.
        """
        from gloss import GLOSS

        call_seed = self._seed * 10000 + self._call_count if self._seed is not None else None
        self._call_count += 1
        gloss = GLOSS(
            space={"candidates": candidates},
            mode="discrete",
            direction="maximize",
            ucb_kappa=self._ucb_kappa,
            diversity_radius=self._diversity_radius,
            diversity_metric=self._diversity_metric,
            local_top_k=self._local_top_k,
            seed=call_seed,
        )
        strategy_points = self._strategy_points or _distribute_points(n_points)

        results = gloss.recommend(
            surrogate=gp,
            X_existing=X_obs,
            strategy_points=strategy_points,
        )

        all_indices = []
        strategies = []
        for r in results:
            if "point" not in r:
                continue
            point = np.array(r["point"])
            dists = np.linalg.norm(candidates - point, axis=1)
            idx = int(np.argmin(dists))
            if idx not in all_indices:
                all_indices.append(idx)
                strategies.append(r.get("strategy", "unknown"))

        return all_indices[:n_points], strategies[:n_points]

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp):
        """Recommend from continuous space.

        Returns:
            (all_points, global_points): all recommended points and global_best-only points.
        """
        from gloss import GLOSS

        call_seed = self._seed * 10000 + self._call_count if self._seed is not None else None
        self._call_count += 1
        gloss = GLOSS(
            space={"bounds": bounds},
            mode="continuous",
            direction="maximize",
            n_random_samples=5000,
            n_top_for_optimization=3,
            ucb_kappa=self._ucb_kappa,
            diversity_radius=self._diversity_radius,
            diversity_metric=self._diversity_metric,
            seed=call_seed,
        )
        strategy_points = self._strategy_points or _distribute_points(n_points)

        results = gloss.recommend(
            surrogate=gp,
            X_existing=X_obs,
            strategy_points=strategy_points,
        )

        all_points = []
        strategies = []
        for r in results:
            if "point" in r:
                all_points.append(r["point"])
                strategies.append(r.get("strategy", "unknown"))

        if len(all_points) == 0:
            fallback_rng = np.random.default_rng(call_seed)
            lows = np.array([b[0] for b in bounds])
            highs = np.array([b[1] for b in bounds])
            fallback = fallback_rng.uniform(lows, highs, size=(n_points, len(bounds)))
            return fallback, ["unknown"] * n_points

        return np.array(all_points[:n_points]), strategies[:n_points]

    def reset(self):
        pass


def _distribute_points(n_points):
    """Distribute n_points across 4 strategies."""
    base = n_points // 4
    remainder = n_points % 4
    counts = [base] * 4
    for i in range(remainder):
        counts[i] += 1
    return {
        "global_best": counts[0],
        "local_best": counts[1],
        "unexplored": counts[2],
        "unconverged": counts[3],
    }


# --------------------------------------------------------------------------
# Bayesian Optimization (Expected Improvement)
# --------------------------------------------------------------------------
class BOAlgorithm:
    """Bayesian Optimization with Expected Improvement."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Recommend from discrete candidates using EI."""
        # Compute EI for all candidates
        mu, sigma = gp.predict(candidates, return_std=True)
        ei = _expected_improvement(mu, sigma, y_best=np.max(y_obs))

        # Exclude already observed points
        obs_set = set(map(tuple, np.round(X_obs, 10)))
        for i in range(len(candidates)):
            if tuple(np.round(candidates[i], 10)) in obs_set:
                ei[i] = -np.inf

        # Select top-n by EI, greedily avoiding duplicates
        indices = []
        order = np.argsort(ei)[::-1]
        for idx in order:
            if len(indices) >= n_points:
                break
            if idx not in indices:
                indices.append(idx)

        return indices

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp):
        """Recommend from continuous space using EI on random samples."""
        ndim = len(bounds)
        rng = self.rng
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])

        # Sample candidates
        samples = rng.uniform(lows, highs, size=(5000, ndim))
        mu, sigma = gp.predict(samples, return_std=True)
        ei = _expected_improvement(mu, sigma, y_best=np.max(y_obs))

        # Refine top candidates with scipy
        from scipy.optimize import minimize

        top_indices = np.argsort(ei)[::-1][:10]
        best_points = []

        for idx in top_indices:
            x0 = samples[idx]

            def neg_ei(x):
                m, s = gp.predict(x.reshape(1, -1), return_std=True)
                return -_expected_improvement(m, s, y_best=np.max(y_obs))[0]

            try:
                res = minimize(
                    neg_ei, x0, method="L-BFGS-B",
                    bounds=[(b[0], b[1]) for b in bounds],
                )
                if res.success:
                    best_points.append(res.x)
                else:
                    best_points.append(x0)
            except Exception:
                best_points.append(x0)

        # Deduplicate and select
        selected = []
        for p in best_points:
            if len(selected) >= n_points:
                break
            is_dup = any(np.linalg.norm(p - s) < 1e-6 for s in selected)
            if not is_dup:
                selected.append(p)

        # Fill with random if not enough
        while len(selected) < n_points:
            selected.append(rng.uniform(lows, highs))

        return np.array(selected[:n_points])

    def reset(self):
        pass


# --------------------------------------------------------------------------
# Genetic Algorithm (with GP-UCB fitness + diversity-aware selection)
# --------------------------------------------------------------------------
class GeneticAlgorithm:
    """Genetic algorithm for active learning using GP-UCB as fitness.

    Strategy:
    - Fitness = GP predicted mean + kappa * GP uncertainty (UCB)
    - Tournament selection from unobserved candidates
    - Niche-sharing diversity: iteratively penalise candidates near
      already-selected points to avoid clustering
    - For continuous spaces: differential-evolution crossover in feature space
      followed by nearest-candidate lookup is not applicable; instead we use
      UCB-optimised random multi-start.
    """

    def __init__(self, kappa=2.0, tournament_size=20, seed=None):
        self.kappa = kappa
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(seed)

    def _fitness(self, gp, X):
        """UCB fitness scores for a batch of points X."""
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Recommend from discrete candidates via tournament selection + niche penalty."""
        fitness = self._fitness(gp, candidates)

        obs_set = set(map(tuple, np.round(X_obs, 10)))
        available = np.array([
            i for i in range(len(candidates))
            if tuple(np.round(candidates[i], 10)) not in obs_set
        ])
        if len(available) == 0:
            return []

        selected = []
        selected_X = []

        avail_fit = fitness[available].copy()

        for _ in range(n_points):
            if len(available) == 0:
                break

            # Niche sharing: penalise candidates close to already-selected
            if selected_X:
                sel = np.array(selected_X)
                # min distance from each available candidate to any selected point
                dists = np.min(
                    np.linalg.norm(
                        candidates[available][:, None, :] - sel[None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                niche_r = np.median(dists) + 1e-10
                penalty = np.exp(-dists / niche_r) * np.std(avail_fit)
                scores = avail_fit - penalty
            else:
                scores = avail_fit

            # Tournament selection
            t_size = min(self.tournament_size, len(available))
            tournament = self.rng.choice(len(available), size=t_size, replace=False)
            winner_local = tournament[np.argmax(scores[tournament])]
            winner_global = available[winner_local]

            selected.append(winner_global)
            selected_X.append(candidates[winner_global].copy())

            # Remove winner from available pool
            mask = np.ones(len(available), dtype=bool)
            mask[winner_local] = False
            available = available[mask]
            avail_fit = avail_fit[mask]

        return selected

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp):
        """Recommend from continuous space via multi-start UCB optimisation."""
        ndim = len(bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])

        from scipy.optimize import minimize as sp_minimize

        # Random population
        pop = self.rng.uniform(lows, highs, size=(500, ndim))
        fitness = self._fitness(gp, pop)
        top_idx = np.argsort(fitness)[::-1][:20]

        def neg_ucb(x):
            mu, sig = gp.predict(x.reshape(1, -1), return_std=True)
            return -(mu[0] + self.kappa * sig[0])

        refined = []
        for idx in top_idx:
            try:
                res = sp_minimize(
                    neg_ucb, pop[idx],
                    method="L-BFGS-B",
                    bounds=[(lo, hi) for lo, hi in zip(lows, highs)],
                )
                refined.append(res.x if res.success else pop[idx])
            except Exception:
                refined.append(pop[idx])

        selected = []
        for p in refined:
            if len(selected) >= n_points:
                break
            if all(np.linalg.norm(p - s) > 1e-6 for s in selected):
                selected.append(p)

        while len(selected) < n_points:
            selected.append(self.rng.uniform(lows, highs))

        return np.array(selected[:n_points])

    def reset(self):
        pass


def _expected_improvement(mu, sigma, y_best):
    """Compute Expected Improvement."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - y_best) / (sigma + 1e-10)
        ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-10] = 0.0
    return ei


# --------------------------------------------------------------------------
# UCB-BO: Bayesian Optimization with Upper Confidence Bound acquisition
# --------------------------------------------------------------------------
class UCBOAlgorithm:
    """Bayesian Optimization with UCB acquisition (mu + kappa*sigma).

    Uses the same fixed GP as BOAlgorithm, but replaces EI with UCB.
    This allows a fair comparison that isolates the effect of multi-strategy
    coordination in GLOSS from the effect of UCB vs EI acquisition.
    """

    def __init__(self, kappa=2.0, seed=None):
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

    def _ucb(self, gp, X):
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Recommend from discrete candidates using UCB."""
        acq = self._ucb(gp, candidates)

        obs_set = set(map(tuple, np.round(X_obs, 10)))
        for i in range(len(candidates)):
            if tuple(np.round(candidates[i], 10)) in obs_set:
                acq[i] = -np.inf

        indices = []
        for idx in np.argsort(acq)[::-1]:
            if len(indices) >= n_points:
                break
            if idx not in indices:
                indices.append(idx)

        return indices

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp):
        """Recommend from continuous space using UCB on random samples + refinement."""
        ndim = len(bounds)
        rng = self.rng
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])

        samples = rng.uniform(lows, highs, size=(5000, ndim))
        acq = self._ucb(gp, samples)

        from scipy.optimize import minimize

        top_indices = np.argsort(acq)[::-1][:10]
        best_points = []

        for idx in top_indices:
            x0 = samples[idx]

            def neg_ucb(x):
                m, s = gp.predict(x.reshape(1, -1), return_std=True)
                return -(m[0] + self.kappa * s[0])

            try:
                res = minimize(
                    neg_ucb, x0, method="L-BFGS-B",
                    bounds=[(b[0], b[1]) for b in bounds],
                )
                best_points.append(res.x if res.success else x0)
            except Exception:
                best_points.append(x0)

        selected = []
        for p in best_points:
            if len(selected) >= n_points:
                break
            if not any(np.linalg.norm(p - s) < 1e-6 for s in selected):
                selected.append(p)

        while len(selected) < n_points:
            selected.append(rng.uniform(lows, highs))

        return np.array(selected[:n_points])

    def reset(self):
        pass


# --------------------------------------------------------------------------
# Random Search
# --------------------------------------------------------------------------
class RandomAlgorithm:
    """Uniform random sampling."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Randomly select from unobserved candidates."""
        obs_set = set(map(tuple, np.round(X_obs, 10)))
        available = [
            i for i in range(len(candidates))
            if tuple(np.round(candidates[i], 10)) not in obs_set
        ]
        if len(available) <= n_points:
            return available
        chosen = self.rng.choice(available, size=n_points, replace=False)
        return chosen.tolist()

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp):
        """Random sampling from bounds."""
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        return self.rng.uniform(lows, highs, size=(n_points, len(bounds)))

    def reset(self):
        pass
