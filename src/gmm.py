"""GMM code."""

from typing import Literal

import numpy as np
from sklearn.mixture import GaussianMixture

type CovarianceType = Literal["full", "tied", "diag", "spherical"]


def select_optimal_components(
    X: np.ndarray,
    max_components: int = 10,
    cv_type: CovarianceType = "full",
    random_state: int = 42,
    reg_covar: float = 1e-4,  # regularize to avoid ill-defined empirical covariance (singularities)
) -> tuple[int, dict]:
    """Try K=1..max_components and keep the lowest BIC."""
    bic_scores = []
    component_range = range(1, max_components + 1)

    for n_components in component_range:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=cv_type,
            random_state=random_state,
            n_init=10,
            max_iter=300,
            reg_covar=reg_covar,
        )
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))

    optimal_components = int(component_range[np.argmin(bic_scores)])

    info = {
        "bic_scores": bic_scores,
        "component_range": list(component_range),
        "optimal": optimal_components,
    }

    return optimal_components, info


def train_class_gmm(
    X_class: np.ndarray,
    n_components: int,
    cv_type: CovarianceType = "full",
    random_state: int = 42,
    reg_covar: float = 1e-4,
) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cv_type,
        random_state=random_state,
        n_init=10,
        max_iter=300,
        warm_start=False,
        reg_covar=reg_covar,
    )
    gmm.fit(X_class)
    return gmm


def train_class_gmms_auto(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_components: int = 10,
    cv_type: CovarianceType = "full",
    random_state: int = 42,
    reg_covar: float = 1e-4,
) -> tuple[dict[int, GaussianMixture], dict]:
    models = {}
    selection_info = {}

    for class_label in [0, 1]:
        class_name = "Benign" if class_label == 0 else "Melanoma"
        X_class = X_train[y_train == class_label]

        print(f"""
try GMM component counts for {class_name}
samples: {len(X_class)}""")

        optimal_n, bic_info = select_optimal_components(
            X_class, max_components, cv_type, random_state + class_label, reg_covar
        )

        print(f"optimal components (BIC): {optimal_n}")

        gmm = train_class_gmm(
            X_class, optimal_n, cv_type, random_state + class_label, reg_covar
        )

        models[class_label] = gmm
        selection_info[class_label] = {
            "n_components": optimal_n,
            "bic_info": bic_info,
            "class_name": class_name,
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
