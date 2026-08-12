import numpy as np
import pytest
from gloss.surrogate.auto_select import auto_select_surrogate


class TestAutoSelect:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.X = rng.uniform(0, 10, (50, 2))
        self.y = np.sin(self.X[:, 0]) + np.cos(self.X[:, 1])

    def test_returns_fitted_model(self):
        model = auto_select_surrogate(self.X, self.y, cv_folds=3)
        preds = model.predict(self.X)
        assert preds.shape == (50,)

    def test_custom_scoring(self):
        model = auto_select_surrogate(
            self.X, self.y, cv_folds=3, scoring="neg_mean_absolute_error"
        )
        preds = model.predict(self.X)
        assert preds.shape == (50,)

    def test_small_dataset_adjusts_folds(self):
        model = auto_select_surrogate(self.X[:6], self.y[:6], cv_folds=5)
        preds = model.predict(self.X[:6])
        assert preds.shape == (6,)
