from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


class _RBFRegressorWrapper(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around scipy.interpolate.RBFInterpolator."""

    def __init__(self, kernel="thin_plate_spline", smoothing=0.0):
        self.kernel = kernel
        self.smoothing = smoothing

    def fit(self, X, y):
        from scipy.interpolate import RBFInterpolator
        self.interpolator_ = RBFInterpolator(
            X, y, kernel=self.kernel, smoothing=self.smoothing
        )
        return self

    def predict(self, X):
        return self.interpolator_(X)


def get_ml_model_configs():
    """Return list of ML model configs for auto-selection.

    Each config is a dict with: name, estimator, param_grid.
    """
    return [
        {
            "name": "GaussianProcess",
            "estimator": GaussianProcessRegressor(random_state=42),
            "param_grid": {
                "kernel": [
                    ConstantKernel() * RBF(),
                    ConstantKernel() * Matern(nu=1.5),
                    ConstantKernel() * Matern(nu=2.5),
                ],
                "alpha": [1e-10, 1e-5, 1e-2],
            },
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
            },
        },
        {
            "name": "XGBoost",
            "estimator": xgb.XGBRegressor(random_state=42, verbosity=0),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.3],
            },
        },
        {
            "name": "LightGBM",
            "estimator": lgb.LGBMRegressor(random_state=42, verbosity=-1),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [-1, 10, 20],
                "learning_rate": [0.01, 0.1, 0.3],
            },
        },
        {
            "name": "SVR",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR()),
            ]),
            "param_grid": {
                "svr__C": [0.1, 1, 10],
                "svr__kernel": ["rbf", "linear"],
                "svr__epsilon": [0.01, 0.1],
            },
        },
        {
            "name": "KNN",
            "estimator": KNeighborsRegressor(),
            "param_grid": {
                "n_neighbors": [3, 5, 10],
                "weights": ["uniform", "distance"],
            },
        },
        {
            "name": "RBF",
            "estimator": _RBFRegressorWrapper(),
            "param_grid": {
                "kernel": ["thin_plate_spline", "multiquadric", "cubic"],
                "smoothing": [0.0, 0.1, 1.0],
            },
        },
    ]
