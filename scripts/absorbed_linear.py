#!/usr/bin/env python3
"""Small weighted OLS implementation with iterative high-dimensional FE absorption."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AbsorbedResult:
    names: list
    beta: np.ndarray
    covariance: np.ndarray
    n: int
    r2_within: float
    iterations: int
    cluster_count: int

    def table(self):
        se = np.sqrt(np.maximum(np.diag(self.covariance), 0.0))
        z = np.divide(self.beta, se, out=np.full_like(self.beta, np.nan), where=se > 0)
        try:
            from scipy.stats import norm

            pvalue = 2.0 * norm.sf(np.abs(z))
        except Exception:
            pvalue = np.full_like(z, np.nan)
        return pd.DataFrame({"term": self.names, "coef": self.beta, "se": se, "z": z, "pvalue": pvalue})

    def contrast(self, weights):
        vector = np.array([float(weights.get(name, 0.0)) for name in self.names])
        estimate = float(vector @ self.beta)
        variance = float(vector @ self.covariance @ vector)
        se = float(np.sqrt(max(variance, 0.0)))
        try:
            from scipy.stats import norm

            pvalue = float(2.0 * norm.sf(abs(estimate / se))) if se > 0 else np.nan
        except Exception:
            pvalue = np.nan
        return estimate, se, pvalue


def _group_codes(series):
    normalized = series.astype(object).where(series.notna(), "Missing").astype(str)
    return pd.factorize(normalized, sort=False)[0].astype(np.int64)


def _demean(matrix, codes, weights):
    groups = int(codes.max()) + 1
    denominator = np.bincount(codes, weights=weights, minlength=groups)
    largest_update = 0.0
    for column in range(matrix.shape[1]):
        numerator = np.bincount(codes, weights=weights * matrix[:, column], minlength=groups)
        means = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        largest_update = max(largest_update, float(np.max(np.abs(means))))
        matrix[:, column] -= means[codes]
    return largest_update


def fit_absorbed_ols(frame, y, x, fixed_effects, weight=None, cluster=None, tolerance=1e-8, max_iter=100):
    required = [y, *x, *fixed_effects]
    if weight:
        required.append(weight)
    if cluster:
        required.append(cluster)
    required = list(dict.fromkeys(required))
    data = frame[required].copy()
    numeric = [y, *x] + ([weight] if weight else [])
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[y, *x])
    if weight:
        data = data[data[weight].gt(0)]
    if len(data) <= len(x) + 2:
        raise ValueError("Too few observations after filtering")

    names = list(x)
    matrix = data[[y, *x]].to_numpy(dtype=np.float64)
    weights = data[weight].to_numpy(dtype=np.float64) if weight else np.ones(len(data), dtype=np.float64)
    codes = []
    for column in fixed_effects:
        if data[column].nunique(dropna=False) <= 1:
            continue
        candidate = _group_codes(data[column])
        if not any(np.array_equal(candidate, existing) for existing in codes):
            codes.append(candidate)

    iterations = 0
    for iteration in range(max_iter):
        change = 0.0
        for group_codes in codes:
            change = max(change, _demean(matrix, group_codes, weights))
        iterations = iteration + 1
        if change < tolerance:
            break

    y_tilde = matrix[:, 0]
    x_tilde = matrix[:, 1:]
    weighted_x = x_tilde * np.sqrt(weights)[:, None]
    weighted_y = y_tilde * np.sqrt(weights)
    keep = np.sqrt(np.sum(weighted_x ** 2, axis=0)) > 1e-10
    if not np.all(keep):
        names = [name for name, use in zip(names, keep) if use]
        x_tilde = x_tilde[:, keep]
        weighted_x = weighted_x[:, keep]
    bread = np.linalg.pinv(weighted_x.T @ weighted_x)
    beta = bread @ (weighted_x.T @ weighted_y)
    residual = y_tilde - x_tilde @ beta
    k = len(beta)

    cluster_count = 0
    if cluster and data[cluster].nunique(dropna=False) >= 10:
        cluster_codes = _group_codes(data[cluster])
        cluster_count = int(cluster_codes.max()) + 1
        score = x_tilde * (weights * residual)[:, None]
        grouped_score = np.column_stack(
            [np.bincount(cluster_codes, weights=score[:, j], minlength=cluster_count) for j in range(k)]
        )
        meat = grouped_score.T @ grouped_score
        correction = (cluster_count / (cluster_count - 1.0)) * ((len(data) - 1.0) / max(len(data) - k, 1.0))
    else:
        score = x_tilde * (weights * residual ** 2)[:, None]
        meat = x_tilde.T @ score
        correction = len(data) / max(len(data) - k, 1.0)
    covariance = correction * bread @ meat @ bread

    denominator = np.sum(weights * y_tilde ** 2)
    r2 = 1.0 - np.sum(weights * residual ** 2) / denominator if denominator > 0 else np.nan
    return AbsorbedResult(names, beta, covariance, len(data), float(r2), iterations, cluster_count)
