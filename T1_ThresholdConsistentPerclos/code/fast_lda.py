"""Exact moment-based re-implementation of sklearn's LinearDiscriminantAnalysis(solver='lsqr',
shrinkage='auto') for classes that are unions of PERCLOS bins (computational shortcut for the LDA grids).

Every T1 grid class (three-class cell, binary-pair cell, single threshold) is a union of the 15 PERCLOS bins
delimited by the 14 cut-points CUTS = {0.20, ..., 0.85} (bin k = #cut-points <= y). Per outer fold the
per-bin sufficient statistics n, sum x, sum x x', sum x^2 x' and sum x^2 (x^2)' are computed once; each
class's mean, within-class standardisation (StandardScaler incl. its constant-feature rule), Ledoit-Wolf
shrinkage of the standardised class data (sklearn.covariance.ledoit_wolf_shrinkage formula, centred), the
prior-weighted pooled covariance, coef = scipy.linalg.lstsq(cov, means.T) and intercept are then obtained
from these moments. Predictions equal sklearn's up to floating-point rounding (checked in
tests/test_t1.py::TestFastLDA on real folds: identical predictions, |proba diff| < 1e-6).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

CUTS = tuple(round(0.20 + 0.05 * k, 2) for k in range(14))


def bin_of(y):
    return np.searchsorted(np.array(CUTS), np.asarray(y, float), side="right")


class BinMoments:
    def __init__(self, X, y, n_bins=15):
        X = np.asarray(X, np.float64)
        b = bin_of(y)
        p = X.shape[1]
        self.p = p
        self.n = np.zeros(n_bins)
        self.s1 = np.zeros((n_bins, p))
        self.S2 = np.zeros((n_bins, p, p))
        self.B = np.zeros((n_bins, p, p))
        self.A = np.zeros((n_bins, p, p))
        for k in range(n_bins):
            Xk = X[b == k]
            if len(Xk) == 0:
                continue
            X2 = Xk * Xk
            self.n[k] = len(Xk)
            self.s1[k] = Xk.sum(0)
            self.S2[k] = Xk.T @ Xk
            self.B[k] = X2.T @ Xk          # B_jk = sum x_j^2 x_k
            self.A[k] = X2.T @ X2          # A_jk = sum x_j^2 x_k^2

    def class_cov(self, bins):
        """sklearn _cov(Xg, 'auto') for the rows in the given bins; returns (mean, cov, n)."""
        bins = list(bins)
        n = self.n[bins].sum()
        s1 = self.s1[bins].sum(0)
        S2 = self.S2[bins].sum(0)
        B = self.B[bins].sum(0)
        A = self.A[bins].sum(0)
        p = self.p
        m = s1 / n
        q = np.diag(S2)
        var = np.maximum(q / n - m * m, 0.0)
        eps = np.finfo(np.float64).eps
        const = var <= n * eps * var + (n * m * eps) ** 2
        s = np.where(const, 1.0, np.sqrt(var))
        # covariance of the class (centred, /n) and of the standardised data
        C = S2 / n - np.outer(m, m)
        R = C / np.outer(s, s)
        R[const, :] = 0.0
        R[:, const] = 0.0
        # sum_i (x_ij - m_j)^2 (x_ik - m_k)^2
        m2 = m * m
        ms1 = m * s1
        M4 = (A - 2 * B * m[None, :] - 2 * B.T * m[:, None] + np.outer(q, m2) + np.outer(m2, q)
              + 4 * np.outer(m, m) * S2 - 2 * np.outer(ms1, m2) - 2 * np.outer(m2, ms1) + n * np.outer(m2, m2))
        M4 = M4 / np.outer(s * s, s * s)
        M4[const, :] = 0.0
        M4[:, const] = 0.0
        trace = np.diag(R).copy()
        mu = trace.sum() / p
        if p == 1:
            shrink = 0.0
        else:
            delta_ = np.sum(R * R)                      # sum (Z'Z)^2 / n^2
            beta_ = M4.sum()
            beta = 1.0 / (p * n) * (beta_ / n - delta_)
            delta = (delta_ - 2.0 * mu * trace.sum() + p * mu ** 2) / p
            beta = min(beta, delta)
            shrink = 0.0 if beta == 0 else beta / delta
        Rs = (1.0 - shrink) * R
        Rs.flat[:: p + 1] += shrink * (np.trace(R) / p)
        cov = s[:, None] * Rs * s[None, :]
        return m, cov, n


class GridLDA:
    """LDA fitted from bin moments. classes: list of bin-lists (class order = label order)."""

    def __init__(self, moments: BinMoments, classes, priors="uniform"):
        means, covs, ns = zip(*[moments.class_cov(c) for c in classes])
        K = len(classes)
        self.priors = np.full(K, 1.0 / K) if priors == "uniform" else np.asarray(ns) / np.sum(ns)
        cov = sum(pr * c for pr, c in zip(self.priors, covs))
        self.means = np.array(means)
        coef = linalg.lstsq(cov, self.means.T)[0].T
        intercept = -0.5 * np.diag(self.means @ coef.T) + np.log(self.priors)
        if K == 2:
            self.coef = (coef[1] - coef[0])[None, :]
            self.intercept = np.array([intercept[1] - intercept[0]])
        else:
            self.coef, self.intercept = coef, intercept
        self.K = K

    def decision_function(self, X):
        d = X @ self.coef.T + self.intercept
        return d[:, 0] if self.K == 2 else d

    def predict(self, X):
        d = self.decision_function(X)
        return (d > 0).astype(int) if self.K == 2 else d.argmax(1)

    def predict_proba(self, X):
        d = self.decision_function(X)
        if self.K == 2:
            p = 1.0 / (1.0 + np.exp(-d))
            return np.stack([1 - p, p], 1)
        d = d - d.max(1, keepdims=True)
        e = np.exp(d)
        return e / e.sum(1, keepdims=True)


def cell_classes(task, cell):
    """Bin lists of the classes of a grid cell (bins 0..14; bin k holds CUTS[k-1] <= y < CUTS[k])."""
    ci = lambda th: CUTS.index(round(th, 2))  # noqa: E731
    if task == "three":
        a, b = ci(cell[0]), ci(cell[1])
        return [list(range(0, a + 1)), list(range(a + 1, b + 1)), list(range(b + 1, 15))]
    if task == "binarypair":
        a, b = ci(cell[0]), ci(cell[1])
        return [list(range(0, a + 1)), list(range(b + 1, 15))]
    if task == "single":
        a = ci(cell)
        return [list(range(0, a + 1)), list(range(a + 1, 15))]
    raise ValueError(task)
