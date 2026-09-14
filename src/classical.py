"""Training-only reference estimators for binary melanoma classification."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)


def _binary_labels(labels: np.ndarray, *, require_both: bool) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("expected nonempty binary training labels")
    if require_both and np.unique(values).size != 2:
        raise ValueError("binary training labels must include both classes")
    return values.astype(int)


def _features(x: np.ndarray, *, dimension: int | None = None) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.isfinite(values).all()
        or (dimension is not None and values.shape[1] != dimension)
    ):
        raise ValueError("expected a nonempty finite feature matrix")
    return values


def fit_prevalence(y_train: np.ndarray) -> dict:
    """Store the observed training prevalence and its majority decision."""
    labels = _binary_labels(y_train, require_both=False)
    prevalence = float(labels.mean())
    return {"prevalence": prevalence, "majority_class": int(prevalence >= 0.5)}


def predict_prevalence(fitted: dict, n_samples: int) -> np.ndarray:
    if not isinstance(n_samples, int) or n_samples < 0:
        raise ValueError("n_samples must be a nonnegative integer")
    return np.full(n_samples, fitted["prevalence"], dtype=float)


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_selection: np.ndarray,
    y_selection: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Select C on selection ROC-AUC; fit the scaler on training rows only."""
    train = _features(x_train)
    selection = _features(x_selection, dimension=train.shape[1])
    labels = _binary_labels(y_train, require_both=True)
    selection_labels = _binary_labels(y_selection, require_both=True)
    if len(train) != len(labels) or len(selection) != len(selection_labels):
        raise ValueError("feature and label row counts differ")

    scaler = StandardScaler().fit(train)
    scaled_train = scaler.transform(train)
    scaled_selection = scaler.transform(selection)
    scores = {}
    best_model = None
    best_auc = -np.inf
    best_c = None
    for c in LOGISTIC_C_GRID:
        candidate = LogisticRegression(
            C=c, random_state=seed, solver="lbfgs", max_iter=1000
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                candidate.fit(scaled_train, labels)
            except ConvergenceWarning as error:
                raise ValueError(f"logistic fit did not converge for C={c}") from error
        probabilities = candidate.predict_proba(scaled_selection)[:, 1]
        auc = float(roc_auc_score(selection_labels, probabilities))
        scores[str(c)] = auc
        if auc > best_auc:
            best_auc, best_c, best_model = auc, c, candidate

    fitted = {"scaler": scaler, "model": best_model}
    info = {
        "selection_metric": "roc_auc",
        "c_grid": list(LOGISTIC_C_GRID),
        "selection_auc_by_c": scores,
        "selected_c": best_c,
        "selected_auc": best_auc,
        "tie_break": "smallest_c",
    }
    return fitted, info


def predict_logistic(fitted: dict, x: np.ndarray) -> np.ndarray:
    matrix = _features(x, dimension=fitted["scaler"].n_features_in_)
    return fitted["model"].predict_proba(fitted["scaler"].transform(matrix))[:, 1]
