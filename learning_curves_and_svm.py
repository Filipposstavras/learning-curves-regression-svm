#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning_curves_and_svm.py
---------------------------------
Clean, reproducible script that implements two problems:

Problem 1 — Regression on synthetic sinusoidal data
  • Generate data: y = 100 * sin(x) + noise
  • Models: Linear Regression, Polynomial Regression (degree=50), Ridge Regression (α=0.001)
  • Learning curves (RMSE) for each model
  • 10-fold Cross-Validation (RMSE)

Problem 2 — SVM classification on Breast Cancer dataset
  • Features: Mean concave points (idx 22) & Worst area (idx 27)
  • Linear SVM with C ∈ {0.1, 1000} — report F1 and #support vectors
  • RBF SVM with GridSearchCV over C and γ — report best params & CV F1
  • Plot decision boundaries for the above

Outputs
  • Figures → ./figures
  • Reports (JSON) → ./reports

Run
  python learning_curves_and_svm.py
Optional
  python learning_curves_and_svm.py --seed 42 --poly_degree 50 --ridge_alpha 0.001
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     learning_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

FIGURES_DIR = "figures"
REPORTS_DIR = "reports"


# ---------- Utils ----------
def safe_makedirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------- Problem 1: Regression ----------
def learning_curve_rmse(
    estimator, X: np.ndarray, y: np.ndarray, seed: int, title: str, out_name: str
) -> Tuple[float, float, str]:
    """Return (train_rmse_last, val_rmse_last, fig_path)."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y.ravel(),
        cv=5,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=seed,
    )
    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
    plt.plot(train_sizes, val_rmse, marker="s", label="Validation RMSE")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title(f"{title} — Learning Curve")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, out_name)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return float(train_rmse[-1]), float(val_rmse[-1]), fig_path


def cv_rmse_mean_std(estimator, X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[float, float]:
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y.ravel(), cv=kf, scoring="neg_root_mean_squared_error")
    return float(-scores.mean()), float(scores.std())


def run_problem1(seed: int, poly_degree: int, ridge_alpha: float) -> Dict:
    # Generate synthetic data
    rng = np.random.default_rng(seed)
    m = 1000
    X = 5 * rng.random((m, 1)) - 2.5  # in [-2.5, 2.5]
    y = np.sin(X) * 100 + rng.normal(0, 1, (m, 1))

    # Standardize X (NOT y)
    scaler_X = StandardScaler()
    X_std = scaler_X.fit_transform(X).astype(float).reshape(-1, 1)

    # Estimators
    lin = LinearRegression()

    poly = make_pipeline(
        PolynomialFeatures(degree=poly_degree, include_bias=True),
        # with_mean=False because PolynomialFeatures can create sparse-like design
        StandardScaler(with_mean=False),
        LinearRegression(),
    )

    ridge = make_pipeline(
        PolynomialFeatures(degree=poly_degree, include_bias=True),
        StandardScaler(with_mean=False),
        Ridge(alpha=ridge_alpha, random_state=seed),
    )

    # Learning curves
    lin_train, lin_val, lc_lin = learning_curve_rmse(lin, X_std, y, seed, "Linear Regression", "p1_linear_lc.png")
    poly_train, poly_val, lc_poly = learning_curve_rmse(poly, X_std, y, seed, f"Polynomial Regression (deg={poly_degree})", "p1_poly_lc.png")
    ridge_train, ridge_val, lc_ridge = learning_curve_rmse(ridge, X_std, y, seed, f"Ridge Regression (α={ridge_alpha})", "p1_ridge_lc.png")

    # 10-fold CV (RMSE)
    lin_cv_mean, lin_cv_std = cv_rmse_mean_std(lin, X_std, y, seed)
    poly_cv_mean, poly_cv_std = cv_rmse_mean_std(poly, X_std, y, seed)
    ridge_cv_mean, ridge_cv_std = cv_rmse_mean_std(ridge, X_std, y, seed)

    return {
        "linear": {
            "learning_curve": {"train_rmse_last": lin_train, "val_rmse_last": lin_val, "figure": lc_lin},
            "cv_rmse": {"mean": lin_cv_mean, "std": lin_cv_std},
        },
        "polynomial": {
            "degree": poly_degree,
            "learning_curve": {"train_rmse_last": poly_train, "val_rmse_last": poly_val, "figure": lc_poly},
            "cv_rmse": {"mean": poly_cv_mean, "std": poly_cv_std},
        },
        "ridge": {
            "alpha": ridge_alpha,
            "learning_curve": {"train_rmse_last": ridge_train, "val_rmse_last": ridge_val, "figure": lc_ridge},
            "cv_rmse": {"mean": ridge_cv_mean, "std": ridge_cv_std},
        },
    }


# ---------- Problem 2: SVM Classification ----------
def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str, out_name: str) -> str:
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=20)
    plt.xlabel("Mean Concave Points (std)")
    plt.ylabel("Worst Area (std)")
    plt.title(title)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, out_name)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def run_problem2(seed: int, c_small: float, c_large: float) -> Dict:
    data = load_breast_cancer()
    # two features: Mean concave points (22), Worst area (27)
    X = data.data[:, [22, 27]].astype(float)
    y = data.target

    # standardize X
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Linear SVM
    svm_c_small = SVC(kernel="linear", C=c_small, random_state=seed)
    svm_c_large = SVC(kernel="linear", C=c_large, random_state=seed)

    svm_c_small.fit(X_std, y)
    svm_c_large.fit(X_std, y)

    f1_small = float(f1_score(y, svm_c_small.predict(X_std)))
    f1_large = float(f1_score(y, svm_c_large.predict(X_std)))

    fig_small = plot_decision_boundary(svm_c_small, X_std, y, f"Linear SVM (C={c_small})", "p2_svm_linear_smallC.png")
    fig_large = plot_decision_boundary(svm_c_large, X_std, y, f"Linear SVM (C={c_large})", "p2_svm_linear_largeC.png")

    # RBF SVM — Grid Search
    param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.1, 1, 10, 100]}
    grid = GridSearchCV(SVC(kernel="rbf", random_state=seed), param_grid, cv=5, scoring="f1")
    grid.fit(X_std, y)
    best_params = grid.best_params_
    best_f1 = float(grid.best_score_)
    best_model = grid.best_estimator_

    fig_rbf = plot_decision_boundary(
        best_model,
        X_std,
        y,
        f"Best RBF SVM (C={best_params['C']}, γ={best_params['gamma']})",
        "p2_svm_rbf_best.png",
    )

    return {
        "linear_svm": {
            "C_small": {"C": c_small, "f1": f1_small, "n_support": list(map(int, svm_c_small.n_support_)), "figure": fig_small},
            "C_large": {"C": c_large, "f1": f1_large, "n_support": list(map(int, svm_c_large.n_support_)), "figure": fig_large},
        },
        "rbf_svm": {"best_params": best_params, "cv_f1_mean": best_f1, "figure": fig_rbf},
    }


# ---------- Main ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learning curves (regression) & SVM classification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--poly_degree", type=int, default=50)
    p.add_argument("--ridge_alpha", type=float, default=0.001)
    p.add_argument("--c_small", type=float, default=0.1)
    p.add_argument("--c_large", type=float, default=1000.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    safe_makedirs(FIGURES_DIR)
    safe_makedirs(REPORTS_DIR)

    p1 = run_problem1(seed=args.seed, poly_degree=args.poly_degree, ridge_alpha=args.ridge_alpha)
    p2 = run_problem2(seed=args.seed, c_small=args.c_small, c_large=args.c_large)

    report = {"problem1_regression": p1, "problem2_svm": p2, "config": vars(args)}
    save_json(report, os.path.join(REPORTS_DIR, "learning_curves_and_svm_report.json"))
    print("Done. Saved:")
    print(" - reports/learning_curves_and_svm_report.json")
    print(" - figures/*.png")


if __name__ == "__main__":
    main()
