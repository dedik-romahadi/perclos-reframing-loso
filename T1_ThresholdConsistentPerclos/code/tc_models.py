"""Train-once distributional models and classical comparators of T1 (PREREG sections 4.3, 4.4, 6).

  LinLN     M1/M16/M18/M20/M22: median imputer -> StandardScaler -> ridge regression of
            z = logit(clip(y, 0.005, 0.995)); logit-normal predictive distribution with homoscedastic
            sigma. alpha in {0.1, 1, 10, 100, 1000, 10000} chosen by inner 5-fold person-grouped CV
            (inner_group_folds(k=5, seed=0)) on the lowest mean over inner-validation persons of the
            person-mean CRPS; sigma^2 = mean squared inner OOF residual (logit scale) at the chosen
            alpha (for every candidate alpha, the CRPS of its OOF predictions uses its own OOF sigma).
  QHGB      M5: 19 quantile HistGradientBoosting regressors (q = 0.05..0.95) on y, monotone
            rearrangement by sorting, piecewise-linear CDF in logit space between knots.
  OrdLogit  M4: cumulative logit P(Y <= k) = sigmoid(b_k - w'x), penalty lambda ||w||^2 on the mean NLL,
            L-BFGS; lambda in {1e-4, 1e-3, 1e-2, 1e-1} by inner 5-fold person-grouped OOF NLL (mean over
            OOF rows). One threshold = L2 logistic regression (binary task).
  LDATS     M3: shrinkage LDA with empirical priors; temperature T in [0.05, 20] minimising the inner
            5-fold OOF NLL of the tempered LDA log-posteriors.
  make_lda  shrinkage LDA (lsqr, Ledoit-Wolf, uniform priors over the classes present) = M2 / R4 protocol.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_expit, logsumexp

import t1_common as C
import seedvig_eog as S

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
QS = tuple(round(0.05 * k, 2) for k in range(1, 20))


class MedianImputer:
    """Column-median imputation fitted on the training rows; identical values to
    SimpleImputer(strategy='median', keep_empty_features=True) (all-NaN column -> 0), but uses np.nanmedian
    (much faster; equality checked in tests/test_t1.py)."""

    def fit(self, X, y=None):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.asarray(X, float), axis=0)
        self.statistics_ = np.where(np.isnan(med), 0.0, med)
        return self

    def transform(self, X):
        X = np.array(X, float, copy=True)
        r, c = np.nonzero(np.isnan(X))
        X[r, c] = self.statistics_[c]
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kw):
        return self


class Prep:
    """MedianImputer -> StandardScaler (training rows only)."""

    def fit(self, X, y=None):
        from sklearn.preprocessing import StandardScaler
        self.imp = MedianImputer().fit(X)
        self.sc = StandardScaler().fit(self.imp.transform(X))
        return self

    def transform(self, X):
        return self.sc.transform(self.imp.transform(X))


def make_prep():
    return Prep()


def make_lda(n_classes: int, priors="uniform"):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    pr = np.full(n_classes, 1.0 / n_classes) if priors == "uniform" else None
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=pr)


# ============================================================================== ridge / LinLN
def ridge_path(X, z, alphas):
    """Ridge coefficients for several alphas (X already standardised). Returns (B (p, A), b0 (A,))."""
    xm = X.mean(0)
    zm = z.mean()
    Xc = X - xm
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uz = U.T @ (z - zm)
    B = np.stack([Vt.T @ (s / (s ** 2 + a) * Uz) for a in alphas], 1)
    b0 = zm - xm @ B
    return B, b0


def person_mean(values, groups):
    return float(np.mean([values[groups == p].mean() for p in np.unique(groups)]))


class LinLN:
    def __init__(self, alphas=ALPHAS, k=5, seed=0):
        self.alphas, self.k, self.seed = tuple(alphas), k, seed

    def fit(self, X, y, groups):
        z = C.zlogit(y)
        n = len(z)
        oof = np.zeros((len(self.alphas), n))
        for tr, va in S.inner_group_folds(groups, k=self.k, seed=self.seed):
            S.assert_no_person_overlap(groups[tr], groups[va])
            prep = make_prep().fit(X[tr])
            B, b0 = ridge_path(prep.transform(X[tr]), z[tr], self.alphas)
            oof[:, va] = (prep.transform(X[va]) @ B + b0).T
        self.inner_sigma = np.sqrt(np.mean((z[None, :] - oof) ** 2, axis=1))
        self.inner_crps = np.array([person_mean(C.ln_crps(oof[a], max(self.inner_sigma[a], C.SIGMA_FLOOR), y), groups)
                                    for a in range(len(self.alphas))])
        self.ia = int(np.argmin(self.inner_crps))
        self.alpha = self.alphas[self.ia]
        self.sigma = float(max(self.inner_sigma[self.ia], C.SIGMA_FLOOR))
        self.prep = make_prep().fit(X)
        B, b0 = ridge_path(self.prep.transform(X), z, (self.alpha,))
        self.coef, self.intercept = B[:, 0], float(b0[0])
        self.oof_mu = oof[self.ia]
        return self

    def predict_mu(self, X):
        return self.prep.transform(X) @ self.coef + self.intercept


def ln_outputs(mu, sigma, y):
    """CDF summaries of a logit-normal model (mu (n,), sigma scalar or (n,))."""
    from scipy.special import expit as _ex
    from scipy.stats import norm
    sigma = np.broadcast_to(np.asarray(sigma, float), np.shape(mu)).astype(float)
    Fcut = C.ln_Fcut(mu, sigma)
    G = C.ln_cdf_grid(mu, sigma)
    mean = C.grid_mean(G)
    median = _ex(mu)
    q = {}
    for lev in (50, 80, 90):
        zq = norm.ppf(0.5 + lev / 200)
        q[lev] = (_ex(mu - zq * sigma), _ex(mu + zq * sigma))
    pit = C.ndtr((C.zlogit(y) - mu) / sigma)
    ls = C.ln_logscore(mu, sigma, y)
    return Fcut, G, mean, median, q, pit, ls


def mixture_ln_outputs(mus, sigmas, y):
    """Equal-weight mixture of logit-normals (M7: mean of the seed CDFs)."""
    from scipy.stats import norm
    mus = [np.asarray(m, float) for m in mus]
    sigmas = [np.broadcast_to(np.asarray(s, float), mus[0].shape) for s in sigmas]
    Fcut = np.mean([C.ln_Fcut(m, s) for m, s in zip(mus, sigmas)], 0)
    G = np.mean([C.ln_cdf_grid(m, s) for m, s in zip(mus, sigmas)], 0)
    mean = C.grid_mean(G)
    median = C.grid_quantile(G, 0.5)
    q = {lev: (C.grid_quantile(G, 0.5 - lev / 200), C.grid_quantile(G, 0.5 + lev / 200)) for lev in (50, 80, 90)}
    z = C.zlogit(y)
    pit = np.mean([C.ndtr((z - m) / s) for m, s in zip(mus, sigmas)], 0)
    dens = np.mean([norm.pdf((z - m) / s) / s for m, s in zip(mus, sigmas)], 0)
    ls = -np.log(np.maximum(dens, 1e-300))
    return Fcut, G, mean, median, q, pit, ls


# ============================================================================== quantile HGB
def _hgbq(q):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.1,
                                         max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False,
                                         random_state=0)


class QHGB:
    def fit(self, X, y):
        self.imp = MedianImputer().fit(X)
        Xi = self.imp.transform(X)
        self.models = [_hgbq(q).fit(Xi, y) for q in QS]
        return self

    def predict_quantiles(self, X):
        Xi = self.imp.transform(X)
        Q = np.stack([m.predict(Xi) for m in self.models], 1)
        return Q


def qhgb_knots(Q):
    """Rearranged (sorted) and clipped quantile knots (n, 19)."""
    Qs = np.sort(Q, axis=1)
    return np.clip(Qs, C.CLIP_LO + 1e-6, C.CLIP_HI - 1e-6)


def qhgb_cdf(knots, x):
    """F(x) for each row: piecewise-linear interpolation of (logit knot, q) with end knots
    (0.005, 0) and (0.995, 1); F = 0 below 0.005 and 1 at/above 0.995. x: (m,) PERCLOS values."""
    from scipy.special import logit
    n = len(knots)
    qs = np.concatenate([[0.0], QS, [1.0]])
    lx = logit(np.clip(np.asarray(x, float), 1e-12, 1 - 1e-12))
    xlo, xhi = C.CLIP_LO, C.CLIP_HI
    out = np.empty((n, len(lx)))
    eps = 1e-9 * np.arange(len(qs))
    for i in range(n):
        lk = logit(np.concatenate([[xlo], knots[i], [xhi]])) + eps
        out[i] = np.interp(lx, lk, qs)
    x = np.asarray(x, float)
    out[:, x < xlo] = 0.0
    out[:, x >= xhi] = 1.0
    return out


def qhgb_outputs(Q, y):
    K = qhgb_knots(Q)
    Fcut = qhgb_cdf(K, np.array(C.CUTS))
    G = qhgb_cdf(K, C.GRID512)
    mean = C.grid_mean(G)
    median = K[:, 9]
    q = {50: (K[:, 4], K[:, 14]), 80: (K[:, 1], K[:, 17]), 90: (K[:, 0], K[:, 18])}
    yc = np.clip(y, C.CLIP_LO, C.CLIP_HI)
    pit = np.array([qhgb_cdf(K[i:i + 1], yc[i:i + 1])[0, 0] for i in range(len(y))])
    return Fcut, G, mean, median, q, pit, K


# ============================================================================== ordinal logit
def _ord_unpack(theta, p, K):
    w = theta[:p]
    b = np.empty(K)
    b[0] = theta[p]
    if K > 1:
        b[1:] = theta[p] + np.cumsum(np.exp(theta[p + 1:p + K]))
    return w, b


def _ord_nll_grad(theta, X, y, lam, K):
    n, p = X.shape
    w, b = _ord_unpack(theta, p, K)
    eta = X @ w
    bb = np.concatenate([[-np.inf], b, [np.inf]])
    up = bb[y + 1] - eta          # b_c - eta
    lo = bb[y] - eta              # b_{c-1} - eta
    Fu = np.where(np.isinf(up), (up > 0).astype(float), expit(np.where(np.isinf(up), 0, up)))
    Fl = np.where(np.isinf(lo), (lo > 0).astype(float), expit(np.where(np.isinf(lo), 0, lo)))
    pr = np.maximum(Fu - Fl, 1e-300)
    fu = np.where(np.isinf(up), 0.0, Fu * (1 - Fu))
    fl = np.where(np.isinf(lo), 0.0, Fl * (1 - Fl))
    # stable log p for the end classes
    logp = np.log(pr)
    m0 = y == 0
    logp[m0] = log_expit(up[m0])
    mK = y == K
    logp[mK] = log_expit(-lo[mK])
    nll = -logp.mean() + lam * w @ w
    g_eta = (fu - fl) / pr
    gw = X.T @ g_eta / n + 2 * lam * w
    gb = np.zeros(K)
    for k in range(K):
        gb[k] = (-np.sum(fu[y == k] / pr[y == k]) + np.sum(fl[y == k + 1] / pr[y == k + 1])) / n
    gth = np.zeros(K)
    gth[0] = gb.sum()
    for k in range(1, K):
        gth[k] = gb[k:].sum() * np.exp(theta[p + k])
    return nll, np.concatenate([gw, gth])


def ord_fit(X, y, lam, K):
    p = X.shape[1]
    prev = np.bincount(y, minlength=K + 1) / len(y)
    cum = np.clip(np.cumsum(prev)[:K], 1e-4, 1 - 1e-4)
    b0 = np.log(cum / (1 - cum))
    th0 = np.concatenate([np.zeros(p), [b0[0]], np.log(np.maximum(np.diff(b0), 1e-3))])
    r = minimize(_ord_nll_grad, th0, args=(X, y, lam, K), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "maxfun": 4000, "gtol": 1e-6})
    return r.x, r


def ord_proba(theta, X, K):
    p = X.shape[1]
    w, b = _ord_unpack(theta, p, K)
    eta = X @ w
    cum = expit(b[None, :] - eta[:, None])
    cum = np.concatenate([np.zeros((len(eta), 1)), cum, np.ones((len(eta), 1))], 1)
    P = np.diff(cum, axis=1)
    return np.clip(P, 1e-15, 1)


class OrdLogit:
    def __init__(self, K, lambdas=LAMBDAS, k=5, seed=0):
        self.K, self.lambdas, self.k, self.seed = K, tuple(lambdas), k, seed

    def fit(self, X, y, groups):
        y = np.asarray(y, int)
        oof_nll = np.zeros(len(self.lambdas))
        for tr, va in S.inner_group_folds(groups, k=self.k, seed=self.seed):
            S.assert_no_person_overlap(groups[tr], groups[va])
            prep = make_prep().fit(X[tr])
            Xt, Xv = prep.transform(X[tr]), prep.transform(X[va])
            for j, lam in enumerate(self.lambdas):
                th, _ = ord_fit(Xt, y[tr], lam, self.K)
                P = ord_proba(th, Xv, self.K)
                oof_nll[j] += -np.sum(np.log(np.clip(P[np.arange(len(va)), y[va]], 1e-12, 1)))
        self.inner_nll = oof_nll / len(y)
        self.lam = self.lambdas[int(np.argmin(self.inner_nll))]
        self.prep = make_prep().fit(X)
        self.theta, self.res = ord_fit(self.prep.transform(X), y, self.lam, self.K)
        return self

    def predict_proba(self, X):
        return ord_proba(self.theta, self.prep.transform(X), self.K)


# ============================================================================== temperature-scaled LDA
def _nll_T(T, L, y):
    Z = L / T
    Z = Z - logsumexp(Z, axis=1, keepdims=True)
    return -np.mean(Z[np.arange(len(y)), y])


class LDATS:
    def __init__(self, k=5, seed=0):
        self.k, self.seed = k, seed

    def fit(self, X, y, groups):
        y = np.asarray(y, int)
        nc = len(np.unique(y))
        L = np.zeros((len(y), nc))
        for tr, va in S.inner_group_folds(groups, k=self.k, seed=self.seed):
            S.assert_no_person_overlap(groups[tr], groups[va])
            prep = make_prep().fit(X[tr])
            m = make_lda(nc, priors="empirical").fit(prep.transform(X[tr]), y[tr])
            L[va] = m.predict_log_proba(prep.transform(X[va]))
        r = minimize_scalar(_nll_T, bounds=(0.05, 20.0), args=(L, y), method="bounded",
                            options={"xatol": 1e-4})
        self.T = float(r.x)
        self.prep = make_prep().fit(X)
        self.lda = make_lda(nc, priors="empirical").fit(self.prep.transform(X), y)
        return self

    def predict_proba(self, X):
        L = self.lda.predict_log_proba(self.prep.transform(X)) / self.T
        L = L - logsumexp(L, axis=1, keepdims=True)
        return np.exp(L)
