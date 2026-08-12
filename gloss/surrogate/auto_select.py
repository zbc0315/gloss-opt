import warnings
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from gloss.surrogate.ml_models import get_ml_model_configs
from gloss.surrogate.nn_models import get_nn_model_configs


def auto_select_surrogate(X, y, cv_folds=5, scoring="neg_root_mean_squared_error"):
    """Select the best surrogate model via cross-validation, retrain on all data.

    Args:
        X: Training features, shape (n_samples, n_features).
        y: Training targets, shape (n_samples,).
        cv_folds: Number of CV folds.
        scoring: sklearn scoring metric string.

    Returns:
        Fitted model with predict(X) method.
    """
    n_samples = X.shape[0]
    actual_folds = min(cv_folds, n_samples)
    if actual_folds < 2:
        actual_folds = 2
    if actual_folds < cv_folds:
        warnings.warn(
            f"Reduced CV folds from {cv_folds} to {actual_folds} due to small dataset "
            f"(n={n_samples})."
        )

    configs = get_ml_model_configs() + get_nn_model_configs(input_dim=X.shape[1])

    best_score = -np.inf
    best_estimator = None
    best_params = None
    best_name = None

    for cfg in configs:
        try:
            gs = GridSearchCV(
                cfg["estimator"],
                cfg["param_grid"],
                cv=actual_folds,
                scoring=scoring,
                refit=False,
                error_score=-np.inf,
            )
            gs.fit(X, y)
            if gs.best_score_ > best_score:
                best_score = gs.best_score_
                best_estimator = cfg["estimator"]
                best_params = gs.best_params_
                best_name = cfg["name"]
        except Exception as e:
            warnings.warn(f"Model {cfg['name']} failed during CV: {e}")
            continue

    if best_estimator is None:
        raise RuntimeError("All models failed during cross-validation.")

    # Clone estimator, set best params, retrain on all data
    best_model = clone(best_estimator).set_params(**best_params)
    best_model.fit(X, y)
    return best_model
