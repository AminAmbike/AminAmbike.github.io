"""
SYDE 572 - Assignment 2
Linear basis function regression on the Boston Housing data.

Every model here is linear in its weights,  y(x, w) = w^T phi(x),
so the only thing that changes between parts is the basis phi:

    polynomial :  phi(x) = [1, x_1, ..., x_D, x_1^2, ..., x_D^2, ..., x_D^M]
    gaussian   :  phi(x) = [1, exp(-||x - mu_j||^2 / (2 s^2)) for j = 1..K]

and whether an L2 penalty  (lambda / 2) ||w||^2  is added to the
sum-of-squares error.

Functions required by the handout (Python names use underscores):
    polynomial_regression          part (a), (b)
    polynomial_regression_1d_vis   part (c)
    polynomial_regression_reg      part (d)
    gaussian_regression            part (e)
    gaussian_regression_reg        part (f)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "housing.data")

FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
N_TRAIN = 100
LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]
RCOND = 1e-10          # singular values below RCOND * s_max are treated as zero


# ------------------------------------------------------------------ data ----
def load_data(path=DATA_FILE):
    """Return X (506 x 13 features) and t (506 median values, in $1000s)."""
    data = np.loadtxt(path)
    return data[:, :13], data[:, 13]


def normalize(X_train, X_other=None):
    """
    Z-score each feature with the mean and standard deviation of the
    TRAINING points only, so nothing about the test set leaks into the model.
    A feature that is constant in the training set (CHAS is 0 for all of the
    first 100 towns) gets sd = 1 so we never divide by zero.
    """
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z_train = (X_train - mu) / sd
    if X_other is None:
        return Z_train, mu, sd
    return Z_train, (X_other - mu) / sd, mu, sd


def train_test_split(n_train=N_TRAIN):
    """First n_train rows are training data, the rest are test data."""
    X, t = load_data()
    return X[:n_train], t[:n_train], X[n_train:], t[n_train:]


# ---------------------------------------------------------------- bases ----
def design_polynomial(Z, degree):
    """
    Monomials of each single variable, no cross terms:
        [1 | Z | Z^2 | ... | Z^degree]
    For D features the matrix has 1 + D * degree columns.
    """
    cols = [np.ones((Z.shape[0], 1))]
    for k in range(1, degree + 1):
        cols.append(Z ** k)
    return np.hstack(cols)


def design_gaussian(Z, centres, s):
    """[1 | exp(-||z - mu_j||^2 / (2 s^2))]  for every centre mu_j."""
    sq = ((Z[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return np.hstack([np.ones((Z.shape[0], 1)), np.exp(-sq / (2.0 * s * s))])


# --------------------------------------------------------------- solvers ----
def fit_least_squares(Phi, t):
    """
    Unregularised solution  w = Phi^+ t  (Moore-Penrose pseudo-inverse).
    When Phi^T Phi is singular (e.g. a constant column, or a feature with
    fewer distinct values than the degree) this returns the minimum-norm
    weight vector instead of failing.
    """
    return np.linalg.pinv(Phi, rcond=RCOND) @ t


def fit_ridge(Phi, t, lam):
    """
    Minimise  1/2 sum (t_n - w^T phi_n)^2 + lam/2 ||w||^2.
    Setting the gradient to zero gives  (lam I' + Phi^T Phi) w = Phi^T t.
    I' is the identity with a 0 in the bias position, so the bias is not
    shrunk towards zero (shrinking it would only pull every prediction
    towards 0 instead of towards the mean house price).
    """
    if lam == 0:
        return fit_least_squares(Phi, t)
    reg = lam * np.eye(Phi.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(reg + Phi.T @ Phi, Phi.T @ t)


def rms(Phi, w, t):
    """Root-mean-square error  sqrt( 1/N sum (y_n - t_n)^2 )."""
    return float(np.sqrt(np.mean((Phi @ w - t) ** 2)))


def kfold_indices(n, k=10):
    """Contiguous folds: fold i validates on rows [i*n/k, (i+1)*n/k)."""
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        val = np.arange(edges[i], edges[i + 1])
        trn = np.setdiff1d(np.arange(n), val)
        yield trn, val


# ============================================================ part (a) ====
def polynomial_regression(degrees=range(1, 8), features=None):
    """
    Fit an unregularised polynomial of each degree on the first 100 towns
    and report training and test RMS error.
    `features` selects columns (default: all 13).
    Returns a dict with the degrees, the two error curves and the weights.
    """
    X_tr, t_tr, X_te, t_te = train_test_split()
    if features is not None:
        X_tr, X_te = X_tr[:, features], X_te[:, features]
    Z_tr, Z_te, mu, sd = normalize(X_tr, X_te)

    out = {"degree": [], "train_rms": [], "test_rms": [], "weights": {}}
    for M in degrees:
        Phi_tr = design_polynomial(Z_tr, M)
        Phi_te = design_polynomial(Z_te, M)
        w = fit_least_squares(Phi_tr, t_tr)
        out["degree"].append(M)
        out["train_rms"].append(rms(Phi_tr, w, t_tr))
        out["test_rms"].append(rms(Phi_te, w, t_te))
        out["weights"][M] = w
    out["mu"], out["sd"] = mu, sd
    return out


# ============================================================ part (c) ====
def polynomial_regression_1d_vis(degree, feature=1, n_grid=400):
    """
    Polynomial regression on ONE feature (default column 2 = ZN, i.e.
    Xn(:,2) in 1-based indexing).  Returns everything needed to plot the
    training points, the test points and the learned curve, in the
    original (un-normalised) units on the x-axis.
    """
    X, t = load_data()
    x = X[:, feature:feature + 1]
    x_tr, x_te, t_tr, t_te = x[:N_TRAIN], x[N_TRAIN:], t[:N_TRAIN], t[N_TRAIN:]
    z_tr, z_te, mu, sd = normalize(x_tr, x_te)

    w = fit_least_squares(design_polynomial(z_tr, degree), t_tr)
    grid = np.linspace(x.min(), x.max(), n_grid)[:, None]
    curve = design_polynomial((grid - mu) / sd, degree) @ w
    return {
        "degree": degree, "feature": FEATURES[feature], "w": w,
        "x_train": x_tr.ravel(), "t_train": t_tr,
        "x_test": x_te.ravel(), "t_test": t_te,
        "grid": grid.ravel(), "curve": curve,
        "train_rms": rms(design_polynomial(z_tr, degree), w, t_tr),
        "test_rms": rms(design_polynomial(z_te, degree), w, t_te),
    }


# ============================================================ part (d) ====
def polynomial_regression_reg(degree=8, feature=1, lambdas=LAMBDAS, k=10):
    """
    L2-regularised degree-8 polynomial on the 2nd feature, first 100 towns.
    10-fold cross-validation picks lambda.  Normalisation is recomputed
    inside every fold from that fold's training rows only.
    """
    X, t = load_data()
    x, tt = X[:N_TRAIN, feature:feature + 1], t[:N_TRAIN]

    val_err = np.zeros((len(lambdas), k))
    trn_err = np.zeros((len(lambdas), k))
    for f, (trn, val) in enumerate(kfold_indices(N_TRAIN, k)):
        z_trn, z_val, _, _ = normalize(x[trn], x[val])
        P_trn = design_polynomial(z_trn, degree)
        P_val = design_polynomial(z_val, degree)
        for i, lam in enumerate(lambdas):
            w = fit_ridge(P_trn, tt[trn], lam)
            trn_err[i, f] = rms(P_trn, w, tt[trn])
            val_err[i, f] = rms(P_val, w, tt[val])

    mean_val = val_err.mean(axis=1)
    best = lambdas[int(np.argmin(mean_val))]
    return {"lambdas": list(lambdas), "val_err": val_err, "trn_err": trn_err,
            "mean_val": mean_val, "mean_trn": trn_err.mean(axis=1), "best_lambda": best}


# ============================================================ part (e) ====
def gaussian_regression(Ks=range(5, 100, 10), s=2.0, seed=572):
    """
    Gaussian RBF regression on all 13 (normalised) features.
    Centres: a random permutation of the 100 training points, then the
    first K of that permutation (so the centre sets are nested in K).
    No regularisation.
    """
    X_tr, t_tr, X_te, t_te = train_test_split()
    Z_tr, Z_te, _, _ = normalize(X_tr, X_te)
    perm = np.random.default_rng(seed).permutation(N_TRAIN)

    out = {"K": [], "train_rms": [], "test_rms": [], "s": s, "seed": seed}
    for K in Ks:
        C = Z_tr[perm[:K]]
        Phi_tr = design_gaussian(Z_tr, C, s)
        Phi_te = design_gaussian(Z_te, C, s)
        w = fit_least_squares(Phi_tr, t_tr)
        out["K"].append(K)
        out["train_rms"].append(rms(Phi_tr, w, t_tr))
        out["test_rms"].append(rms(Phi_te, w, t_te))
    return out


# ============================================================ part (f) ====
def gaussian_regression_reg(K=90, s=2.0, lambdas=LAMBDAS, k=10, seed=572):
    """
    L2-regularised Gaussian RBF regression, all features, first 100 towns,
    K = 90 basis functions, lambda chosen by 10-fold cross-validation.
    In each fold the K centres are a random permutation of that fold's
    training rows (a fold has exactly 90 training rows, so every one of
    them becomes a centre) and the validation rows are never used as centres.
    """
    X, t = load_data()
    Xa, ta = X[:N_TRAIN], t[:N_TRAIN]
    rng = np.random.default_rng(seed)

    val_err = np.zeros((len(lambdas), k))
    trn_err = np.zeros((len(lambdas), k))
    for f, (trn, val) in enumerate(kfold_indices(N_TRAIN, k)):
        Z_trn, Z_val, _, _ = normalize(Xa[trn], Xa[val])
        C = Z_trn[rng.permutation(len(trn))[:K]]
        P_trn = design_gaussian(Z_trn, C, s)
        P_val = design_gaussian(Z_val, C, s)
        for i, lam in enumerate(lambdas):
            w = fit_ridge(P_trn, ta[trn], lam)
            trn_err[i, f] = rms(P_trn, w, ta[trn])
            val_err[i, f] = rms(P_val, w, ta[val])

    mean_val = val_err.mean(axis=1)
    best = lambdas[int(np.argmin(mean_val))]
    return {"lambdas": list(lambdas), "val_err": val_err, "trn_err": trn_err,
            "mean_val": mean_val, "mean_trn": trn_err.mean(axis=1), "best_lambda": best}


if __name__ == "__main__":
    a = polynomial_regression()
    for M, tr, te in zip(a["degree"], a["train_rms"], a["test_rms"]):
        print(f"degree {M}:  train RMS {tr:8.3f}   test RMS {te:.4g}")
    d = polynomial_regression_reg()
    print("poly reg   mean validation RMS:", np.round(d["mean_val"], 3), " best lambda:", d["best_lambda"])
    e = gaussian_regression()
    for K, tr, te in zip(e["K"], e["train_rms"], e["test_rms"]):
        print(f"K = {K:2d}:  train RMS {tr:6.3f}   test RMS {te:6.3f}")
    g = gaussian_regression_reg()
    print("gauss reg  mean validation RMS:", np.round(g["mean_val"], 3), " best lambda:", g["best_lambda"])
