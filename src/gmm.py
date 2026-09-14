"""Class-conditional Gaussian mixture models with explicit fit diagnostics."""

import warnings
from typing import Literal

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

type CovarianceType = Literal["full", "tied", "diag", "spherical"]


def _validate_settings(
    x: np.ndarray,
    n_components: int,
    cv_type: CovarianceType,
    reg_covar: float,
    max_iter: int,
    *,
    component_name: str,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("GMM requires a nonempty finite feature matrix")
    if (
        isinstance(n_components, bool)
        or not isinstance(n_components, int)
        or n_components < 1
    ):
        raise ValueError(f"{component_name} must be a positive integer")
    if component_name == "n_components" and n_components > len(values):
        raise ValueError("n_components exceeds available class samples")
    if cv_type not in ("full", "tied", "diag", "spherical"):
        raise ValueError("invalid GMM covariance_type")
    if not np.isfinite(reg_covar) or reg_covar <= 0:
        raise ValueError("reg_covar must be positive and finite")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    return values


def _fit_gmm(x: np.ndarray, **kwargs) -> GaussianMixture:
    model = GaussianMixture(**kwargs)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x)
    model.fit_warnings_ = list(
        dict.fromkeys(
            str(item.message)
            for item in captured
            if issubclass(item.category, ConvergenceWarning)
        )
    )
    for item in captured:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn(item.message, item.category, stacklevel=2)
    if not model.converged_:
        raise ValueError(f"GMM did not converge for n_components={model.n_components}")
    return model


def select_optimal_components(
    X: np.ndarray,
    max_components: int = 10,
    cv_type: CovarianceType = "full",
    random_state: int = 42,
    reg_covar: float = 1e-4,
    max_iter: int = 300,
) -> tuple[int, dict]:
    """Try feasible K on training rows and keep the lowest BIC."""
    x = _validate_settings(
        X,
        max_components,
        cv_type,
        reg_covar,
        max_iter,
        component_name="max_components",
    )
    effective_max = min(max_components, len(x))
    bic_scores = []
    initialization_warnings = {}
    component_range = range(1, effective_max + 1)

    for n_components in component_range:
        gmm = _fit_gmm(
            x,
            n_components=n_components,
            covariance_type=cv_type,
            random_state=random_state,
            n_init=10,
            max_iter=max_iter,
            reg_covar=reg_covar,
        )
        bic_scores.append(float(gmm.bic(x)))
        initialization_warnings[str(n_components)] = gmm.fit_warnings_

    optimal_components = int(component_range[np.argmin(bic_scores)])

    info = {
        "bic_scores": bic_scores,
        "component_range": list(component_range),
        "optimal": optimal_components,
        "requested_max_components": max_components,
        "effective_max_components": effective_max,
        "reg_covar": reg_covar,
        "covariance_type": cv_type,
        "max_iter": max_iter,
        "initialization_warnings": initialization_warnings,
    }

    return optimal_components, info


def train_class_gmm(
    X_class: np.ndarray,
    n_components: int,
    cv_type: CovarianceType = "full",
    random_state: int = 42,
    reg_covar: float = 1e-4,
    max_iter: int = 300,
) -> GaussianMixture:
    x = _validate_settings(
        X_class,
        n_components,
        cv_type,
        reg_covar,
        max_iter,
        component_name="n_components",
    )
    return _fit_gmm(
        x,
        n_components=n_components,
        covariance_type=cv_type,
        random_state=random_state,
        n_init=10,
        max_iter=max_iter,
        warm_start=False,
        reg_covar=reg_covar,
    )


def train_class_gmms_auto(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_components: int = 10,
    cv_type: CovarianceType = "full",
    random_state: int = 42,
    reg_covar: float = 1e-4,
    max_iter: int = 300,
) -> tuple[dict[int, GaussianMixture], dict]:
    x = _validate_settings(
        X_train,
        max_components,
        cv_type,
        reg_covar,
        max_iter,
        component_name="max_components",
    )
    y = np.asarray(y_train)
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
        raise ValueError("GMM requires aligned binary training labels")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("GMM requires both training classes")
    models = {}
    selection_info = {}

    for class_label in [0, 1]:
        class_name = "Benign" if class_label == 0 else "Melanoma"
        X_class = x[y == class_label]

        print(f"""
try GMM component counts for {class_name}
samples: {len(X_class)}""")

        optimal_n, bic_info = select_optimal_components(
            X_class,
            max_components,
            cv_type,
            random_state + class_label,
            reg_covar,
            max_iter,
        )

        print(f"optimal components (BIC): {optimal_n}")

        gmm = train_class_gmm(
            X_class,
            optimal_n,
            cv_type,
            random_state + class_label,
            reg_covar,
            max_iter,
        )

        models[class_label] = gmm
        selection_info[class_label] = {
            "n_components": optimal_n,
            "bic_info": bic_info,
            "class_name": class_name,
            "requested_max_components": max_components,
            "effective_max_components": bic_info["effective_max_components"],
            "converged": bool(gmm.converged_),
            "n_iter": int(gmm.n_iter_),
            "reg_covar": reg_covar,
            "covariance_type": cv_type,
            "max_iter": max_iter,
            "fit_warnings": gmm.fit_warnings_,
        }

    return models, selection_info


def compute_log_likelihood(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    return gmm.score_samples(X)


def compute_class_likelihoods(
    models: dict[int, GaussianMixture], X: np.ndarray
) -> np.ndarray:
    n_samples = X.shape[0]
    n_classes = len(models)

    likelihoods = np.zeros((n_samples, n_classes))

    for class_label in sorted(models.keys()):
        likelihoods[:, class_label] = compute_log_likelihood(models[class_label], X)

    return likelihoods


def print_gmm_summary(selection_info: dict) -> None:
    lines = ["", "GMM summary"]

    for class_label in sorted(selection_info.keys()):
        info = selection_info[class_label]
        lines.append(f"{info['class_name']}: K={info['n_components']}")
        lines.append(
            f"  first BIC values: {[round(x, 1) for x in info['bic_info']['bic_scores'][:5]]}"
        )

    print("\n".join(lines))
