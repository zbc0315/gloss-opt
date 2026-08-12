import numpy as np
import pytest
from gloss.surrogate.ml_models import get_ml_model_configs


class TestMLModels:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.X = rng.uniform(0, 10, (50, 2))
        self.y = np.sin(self.X[:, 0]) + np.cos(self.X[:, 1])

    def test_get_configs_returns_list(self):
        configs = get_ml_model_configs()
        assert len(configs) > 0
        for cfg in configs:
            assert "name" in cfg
            assert "estimator" in cfg
            assert "param_grid" in cfg

    def test_each_model_fits_and_predicts(self):
        configs = get_ml_model_configs()
        for cfg in configs:
            model = cfg["estimator"]
            model.fit(self.X, self.y)
            preds = model.predict(self.X)
            assert preds.shape == (50,), f"{cfg['name']} predict shape wrong"
