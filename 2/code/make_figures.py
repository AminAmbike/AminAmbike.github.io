"""
SYDE 572 - Assignment 2
Runs every experiment, saves every figure on the project page to ../media/
and writes results.json with all the numbers quoted in the write-up.

    python3 make_figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

from regression import (FEATURES, LAMBDAS, N_TRAIN, load_data, normalize,
                        design_polynomial, design_gaussian, fit_least_squares,
                        fit_ridge, rms, polynomial_regression,
                        polynomial_regression_1d_vis, polynomial_regression_reg,
                        gaussian_regression, gaussian_regression_reg)

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "media")
os.makedirs(MEDIA, exist_ok=True)

# ---------------------------------------------------------------- style ----
INK = "#1d2330"
MUTED = "#6b6f7a"
GRID = "#e6e4df"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"      # training
ORANGE = "#eb6834"    # test
VIOLET = "#4a3aa7"    # validation / model
AQUA = "#1baf7a"      # chosen value
RED = "#e34948"
LIGHT = "#c9c6bf"

plt.rcParams.update({
    "font.family": ["Inter", "Avenir Next", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelcolor": INK,
    "axes.edgecolor": LIGHT,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
})


def save(fig, name):
    fig.savefig(os.path.join(MEDIA, name))
    plt.close(fig)
    print("saved", name)


def fmt_big(v):
    """Readable label for errors that range from 1 to 10^16."""
    if v < 100:
        return f"{v:.2f}"
    if v < 1e4:
        return f"{v:,.0f}"
    e = int(np.floor(np.log10(v)))
    return f"{v / 10**e:.1f}e{e}"


X_ALL, T_ALL = load_data()
RESULTS = {}

# ========================================================= part (a) =======
a = polynomial_regression()
RESULTS["a"] = {"degree": a["degree"], "train_rms": a["train_rms"], "test_rms": a["test_rms"]}

# For context: the same experiment with 100 RANDOM training towns
# (50 different random splits, median and inter-quartile range).
rand_tr, rand_te = [], []
for seed in range(50):
    p = np.random.default_rng(seed).permutation(len(T_ALL))
    tr, te = p[:N_TRAIN], p[N_TRAIN:]
    Ztr, Zte, _, _ = normalize(X_ALL[tr], X_ALL[te])
    r_tr, r_te = [], []
    for M in a["degree"]:
        w = fit_least_squares(design_polynomial(Ztr, M), T_ALL[tr])
        r_tr.append(rms(design_polynomial(Ztr, M), w, T_ALL[tr]))
        r_te.append(rms(design_polynomial(Zte, M), w, T_ALL[te]))
    rand_tr.append(r_tr)
    rand_te.append(r_te)
rand_tr, rand_te = np.array(rand_tr), np.array(rand_te)
RESULTS["a_random_split"] = {
    "train_median": np.median(rand_tr, 0).tolist(),
    "test_median": np.median(rand_te, 0).tolist(),
    "test_q25": np.percentile(rand_te, 25, 0).tolist(),
    "test_q75": np.percentile(rand_te, 75, 0).tolist(),
}

# how far outside the training range the test towns are, per feature
Z_tr, Z_te, mu, sd = normalize(X_ALL[:N_TRAIN], X_ALL[N_TRAIN:])
RESULTS["test_max_abs_z"] = dict(zip(FEATURES, np.abs(Z_te).max(0).round(2).tolist()))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"wspace": 0.28})
ax = axes[0]
deg = a["degree"]
ax.semilogy(deg, a["train_rms"], "-o", color=BLUE, lw=2, ms=6, label="training RMS (first 100 towns)")
ax.semilogy(deg, a["test_rms"], "-o", color=ORANGE, lw=2, ms=6, label="test RMS (remaining 406 towns)")
for M, v in zip(deg, a["test_rms"]):
    ax.annotate(fmt_big(v), (M, v), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8.5, color=ORANGE)
for M, v in zip(deg, a["train_rms"]):
    ax.annotate(f"{v:.2f}", (M, v), textcoords="offset points", xytext=(0, -15),
                ha="center", fontsize=8.5, color=BLUE)
ax.axhline(np.std(T_ALL[N_TRAIN:]), color=MUTED, lw=1, ls="--")
ax.text(7.35, np.std(T_ALL[N_TRAIN:]) * 1.35, "predict-the-mean\nbaseline (9.83)", fontsize=8.5,
        color=MUTED, ha="right")
ax.set_xlabel("polynomial degree M")
ax.set_ylabel("RMS error ($1000s, log scale)")
ax.set_title("(a) Train vs test error, specified split")
ax.set_ylim(0.03, 1e18)
ax.set_xticks(deg)
ax.legend(loc="upper left")

ax = axes[1]
ax.fill_between(deg, RESULTS["a_random_split"]["test_q25"], RESULTS["a_random_split"]["test_q75"],
                color=ORANGE, alpha=0.15, lw=0, label="test RMS, inter-quartile range")
ax.semilogy(deg, RESULTS["a_random_split"]["train_median"], "-o", color=BLUE, lw=2, ms=6,
            label="training RMS (median)")
ax.semilogy(deg, RESULTS["a_random_split"]["test_median"], "-o", color=ORANGE, lw=2, ms=6,
            label="test RMS (median)")
best = int(np.argmin(RESULTS["a_random_split"]["test_median"]))
ax.scatter([deg[best]], [RESULTS["a_random_split"]["test_median"][best]], s=170,
           facecolor="none", edgecolor=AQUA, lw=2, zorder=5)
ax.annotate(f"lowest test error\nM = {deg[best]}", (deg[best], RESULTS["a_random_split"]["test_median"][best]),
            textcoords="offset points", xytext=(10, 42), ha="left", fontsize=8.5, color=AQUA)
ax.set_xlabel("polynomial degree M")
ax.set_ylabel("RMS error ($1000s, log scale)")
ax.set_title("Context: 100 random training towns (50 splits)")
ax.set_xticks(deg)
ax.legend(loc="upper left")
save(fig, "a_poly_errors.png")

# ========================================================= part (b) =======
w1 = a["weights"][1]
RESULTS["b"] = {
    "bias": float(w1[0]),
    "weights": {n: float(v) for n, v in zip(FEATURES, w1[1:])},
    "per_unit": {n: float(v / s) for n, v, s in zip(FEATURES, w1[1:], a["sd"])},
    "train_mean": dict(zip(FEATURES, a["mu"].tolist())),
    "train_sd": dict(zip(FEATURES, a["sd"].tolist())),
    "train_corr_with_medv": {n: (float(np.corrcoef(X_ALL[:N_TRAIN, i], T_ALL[:N_TRAIN])[0, 1])
                                 if a["sd"][i] != 1.0 or FEATURES[i] != "CHAS" else 0.0)
                             for i, n in enumerate(FEATURES)},
}
# the same degree-1 model fitted on all 506 towns, for comparison
Z_all, _, _ = normalize(X_ALL)
w_all = fit_least_squares(design_polynomial(Z_all, 1), T_ALL)
RESULTS["b"]["all_506_weights"] = {n: float(v) for n, v in zip(FEATURES, w_all[1:])}

fig, ax = plt.subplots(figsize=(8.6, 4.6))
order = np.arange(len(FEATURES))[::-1]
vals = w1[1:]
colors = [AQUA if FEATURES[i] in ("RM", "DIS") else (BLUE if vals[i] >= 0 else ORANGE) for i in range(13)]
ax.barh(order, vals, color=colors, height=0.62)
for y, i in zip(order, range(13)):
    v = vals[i]
    ax.text(v + (0.08 if v >= 0 else -0.08), y, f"{v:+.3f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8.8, color=INK)
ax.set_yticks(order)
ax.set_yticklabels([f"{n}  (feature {i + 1})" for i, n in enumerate(FEATURES)])
ax.axvline(0, color=MUTED, lw=0.9)
ax.set_xlim(-3.2, 5.6)
ax.set_xlabel("weight ($1000s per one training standard deviation of the feature)")
ax.set_title(f"(b) Degree-1 weights  (bias w0 = {w1[0]:.3f})")
ax.grid(axis="y", visible=False)
save(fig, "b_weights.png")

# ========================================================= part (c) =======
RESULTS["c"] = {}
vis = {}
for M in range(1, 11):
    v = polynomial_regression_1d_vis(M)
    vis[M] = v
    RESULTS["c"][M] = {"train_rms": v["train_rms"], "test_rms": v["test_rms"],
                       "curve_min": float(v["curve"].min()), "curve_max": float(v["curve"].max())}
ZN_TRAIN_VALUES = np.unique(X_ALL[:N_TRAIN, 1])
RESULTS["c_zn_train_values"] = ZN_TRAIN_VALUES.tolist()
RESULTS["c_zn_test_values"] = np.unique(X_ALL[N_TRAIN:, 1]).tolist()
RESULTS["c_zn_zero_fraction_train"] = float(np.mean(X_ALL[:N_TRAIN, 1] == 0))


def vis_plot(ax, v, ylim=(-5, 60), legend=True, small=False):
    ax.scatter(v["x_test"], v["t_test"], s=14 if not small else 6, color=ORANGE, alpha=0.35,
               lw=0, label="test towns")
    ax.scatter(v["x_train"], v["t_train"], s=26 if not small else 10, color=BLUE, alpha=0.85,
               edgecolor="white", lw=0.5, label="training towns")
    y = np.clip(v["curve"], ylim[0] - 50, ylim[1] + 50)
    ax.plot(v["grid"], y, color=VIOLET, lw=2.2 if not small else 1.6, label=f"degree-{v['degree']} fit")
    for z in ZN_TRAIN_VALUES:
        ax.axvline(z, color=GRID, lw=0.8, zorder=0)
    ax.set_ylim(*ylim)
    ax.set_xlim(-3, 103)
    if legend:
        ax.legend(loc="upper left", ncol=3)


for M in (1, 4, 10):
    v = vis[M]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    vis_plot(ax, v)
    ax.set_xlabel("ZN  (% of residential land zoned for lots > 25,000 sq ft)")
    ax.set_ylabel("MEDV  ($1000s)")
    ax.set_title(f"(c) Degree {M}:  train RMS {v['train_rms']:.2f},  test RMS {v['test_rms']:.2f}")
    if v["curve"].min() < -5 or v["curve"].max() > 60:
        ax.text(0.98, 0.04, f"curve range {v['curve'].min():.0f} to {v['curve'].max():.0f}\n(clipped to the axes)",
                transform=ax.transAxes, ha="right", fontsize=8.5, color=VIOLET)
    save(fig, f"c_deg{M}.png")

fig, axes = plt.subplots(2, 5, figsize=(14, 5.4), sharex=True, sharey=True,
                         gridspec_kw={"wspace": 0.08, "hspace": 0.32})
for M, ax in zip(range(1, 11), axes.ravel()):
    v = vis[M]
    vis_plot(ax, v, legend=False, small=True)
    ax.set_title(f"M = {M}   test {fmt_big(v['test_rms'])}", fontsize=10.5)
for ax in axes[1]:
    ax.set_xlabel("ZN")
for ax in axes[:, 0]:
    ax.set_ylabel("MEDV")
save(fig, "c_gallery.png")

# ========================================================= part (d) =======
d = polynomial_regression_reg()
RESULTS["d"] = {"lambdas": d["lambdas"], "mean_val": d["mean_val"].tolist(),
                "mean_trn": d["mean_trn"].tolist(), "val_err": d["val_err"].tolist(),
                "best_lambda": d["best_lambda"],
                "median_val": np.median(d["val_err"], 1).tolist()}
RESULTS["d_fold6_zn"] = X_ALL[50:60, 1].tolist()

# refit on all 100 towns with the chosen lambda and score the 406 test towns
x_tr, x_te = X_ALL[:N_TRAIN, 1:2], X_ALL[N_TRAIN:, 1:2]
z_tr, z_te, _, _ = normalize(x_tr, x_te)
P_tr, P_te = design_polynomial(z_tr, 8), design_polynomial(z_te, 8)
RESULTS["d_test"] = {}
for lam in LAMBDAS:
    w = fit_ridge(P_tr, T_ALL[:N_TRAIN], lam)
    RESULTS["d_test"][str(lam)] = {"train_rms": rms(P_tr, w, T_ALL[:N_TRAIN]),
                                   "test_rms": rms(P_te, w, T_ALL[N_TRAIN:])}


def cv_plot(res, title, fname, note_fold=None, best_xy=(14, -38)):
    lams = np.array(res["lambdas"][1:], dtype=float)
    vals = res["val_err"][1:]
    mean_v = res["mean_val"][1:]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for f in range(vals.shape[1]):
        ax.semilogx(lams, vals[:, f], color=VIOLET, alpha=0.16, lw=1, zorder=1)
    ax.semilogx([], [], color=VIOLET, alpha=0.3, lw=1, label="individual folds")
    ax.semilogx(lams, res["mean_trn"][1:], "-o", color=BLUE, lw=1.6, ms=5, label="mean training RMS")
    ax.semilogx(lams, mean_v, "-o", color=VIOLET, lw=2.4, ms=7, label="mean validation RMS (10 folds)")
    base = res["mean_val"][0]
    ax.axhline(base, color=RED, ls="--", lw=1.4, label=f"λ = 0 baseline: {fmt_big(base)}")
    i = int(np.argmin(mean_v))
    ax.scatter([lams[i]], [mean_v[i]], s=220, facecolor="none", edgecolor=AQUA, lw=2.2, zorder=6)
    ax.annotate(f"best λ = {lams[i]:g}\nmean val RMS = {mean_v[i]:.2f}", (lams[i], mean_v[i]),
                textcoords="offset points", xytext=best_xy, fontsize=9.5, color=AQUA, weight="semibold")
    for lam, v in zip(lams, mean_v):
        if lam != lams[i]:
            ax.annotate(fmt_big(v), (lam, v), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=VIOLET)
    ax.set_xlabel("regularising constant λ (log scale)")
    ax.set_ylabel("RMS error ($1000s)")
    ax.set_title(title)
    ax.xaxis.set_major_locator(FixedLocator(lams))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{l:g}" for l in lams]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    if note_fold is not None:
        note_fold(ax)
    ax.legend(loc="upper left", fontsize=9)
    return fig, ax


def d_extra(ax):
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3e7)
    ax.set_ylabel("RMS error ($1000s, log scale)")
    v6 = d["val_err"][1:, 5]
    ax.annotate("fold 6 (towns 51 to 60)", (0.01, v6[0]), textcoords="offset points", xytext=(30, 6),
                fontsize=8.5, color=VIOLET, alpha=0.8)


fig, ax = cv_plot(d, "(d) Degree-8 polynomial on ZN: 10-fold cross-validation", "", d_extra)
save(fig, "d_poly_reg_cv.png")

# ========================================================= part (e) =======
e = gaussian_regression()
RESULTS["e"] = {"K": e["K"], "train_rms": e["train_rms"], "test_rms": e["test_rms"],
                "s": e["s"], "seed": e["seed"]}
# repeat with 30 other random permutations to see how much the centres matter
reps = [gaussian_regression(seed=s) for s in range(30)]
rep_tr = np.array([r["train_rms"] for r in reps])
rep_te = np.array([r["test_rms"] for r in reps])
RESULTS["e_repeats"] = {"train_median": np.median(rep_tr, 0).tolist(),
                        "test_median": np.median(rep_te, 0).tolist(),
                        "test_q10": np.percentile(rep_te, 10, 0).tolist(),
                        "test_q90": np.percentile(rep_te, 90, 0).tolist()}
# width sensitivity
RESULTS["e_width"] = {str(s): gaussian_regression(s=s)["test_rms"] for s in (1.0, 2.0, 4.0)}
# how strongly test towns activate their nearest centre (all 100 training points as centres)
sq_te = ((Z_te[:, None, :] - Z_tr[None, :, :]) ** 2).sum(2)
act_te = np.exp(-sq_te / (2 * 2.0 ** 2)).max(1)
sq_tr = ((Z_tr[:, None, :] - Z_tr[None, :, :]) ** 2).sum(2)
np.fill_diagonal(sq_tr, np.inf)
act_tr = np.exp(-sq_tr / (2 * 2.0 ** 2)).max(1)
RESULTS["e_activation"] = {"test_median": float(np.median(act_te)),
                           "test_frac_below_0.01": float(np.mean(act_te < 0.01)),
                           "train_loo_median": float(np.median(act_tr))}
mean_baseline = float(np.sqrt(np.mean((T_ALL[N_TRAIN:] - T_ALL[:N_TRAIN].mean()) ** 2)))
RESULTS["mean_baseline_test_rms"] = mean_baseline

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"wspace": 0.25, "width_ratios": [1.35, 1]})
ax = axes[0]
K = e["K"]
ax.fill_between(K, RESULTS["e_repeats"]["test_q10"], RESULTS["e_repeats"]["test_q90"], color=ORANGE,
                alpha=0.13, lw=0, label="test RMS, 10th to 90th pct. over 30 centre draws")
ax.plot(K, e["train_rms"], "-o", color=BLUE, lw=2, ms=6, label="training RMS")
ax.plot(K, e["test_rms"], "-o", color=ORANGE, lw=2, ms=6, label="test RMS")
ax.axhline(mean_baseline, color=MUTED, ls="--", lw=1)
ax.text(96, mean_baseline - 1.6, f"predict the training mean: {mean_baseline:.2f}", ha="right",
        fontsize=8.5, color=MUTED)
for k_, v in zip(K, e["train_rms"]):
    ax.annotate(f"{v:.2f}", (k_, v), textcoords="offset points", xytext=(0, -15), ha="center",
                fontsize=8.3, color=BLUE)
for k_, v in zip(K, e["test_rms"]):
    ax.annotate(f"{v:.1f}", (k_, v), textcoords="offset points", xytext=(0, 8), ha="center",
                fontsize=8.3, color=ORANGE)
ax.set_xticks(K)
ax.set_ylim(-1.6, 21)
ax.set_xlabel("number of Gaussian basis functions K")
ax.set_ylabel("RMS error ($1000s)")
ax.set_title(f"(e) Gaussian basis, s = {e['s']:g}, no regularisation")
ax.legend(loc="upper left", fontsize=9)

ax = axes[1]
bins = np.linspace(0, 1, 26)
ax.hist(act_te, bins=bins, color=ORANGE, alpha=0.6, label="test town")
ax.hist(act_tr, bins=bins, color=BLUE, alpha=0.75, label="training town (nearest other centre)")
ax.set_xlabel(r"largest basis activation  $\max_j\ \phi_j(\mathbf{x})$")
ax.set_ylabel("number of towns")
ax.set_title("Why the test error stays high")
ax.legend(loc="upper center", fontsize=9)
save(fig, "e_gauss_errors.png")

# ========================================================= part (f) =======
f = gaussian_regression_reg()
RESULTS["f"] = {"lambdas": f["lambdas"], "mean_val": f["mean_val"].tolist(),
                "mean_trn": f["mean_trn"].tolist(), "val_err": f["val_err"].tolist(),
                "best_lambda": f["best_lambda"]}
# refit on all 100 towns: centres = first 90 of a random permutation of the 100
Ztr, Zte, _, _ = normalize(X_ALL[:N_TRAIN], X_ALL[N_TRAIN:])
C = Ztr[np.random.default_rng(572).permutation(N_TRAIN)[:90]]
P_tr, P_te = design_gaussian(Ztr, C, 2.0), design_gaussian(Zte, C, 2.0)
RESULTS["f_test"] = {}
for lam in LAMBDAS:
    w = fit_ridge(P_tr, T_ALL[:N_TRAIN], lam)
    RESULTS["f_test"][str(lam)] = {"train_rms": rms(P_tr, w, T_ALL[:N_TRAIN]),
                                   "test_rms": rms(P_te, w, T_ALL[N_TRAIN:])}


def f_extra(ax):
    ax.set_ylim(0, 13.5)


fig, ax = cv_plot(f, "(f) Gaussian basis, K = 90, all features: 10-fold cross-validation", "", f_extra, best_xy=(22, 34))
save(fig, "f_gauss_reg_cv.png")

# ------------------------------------------------------------------ dump ----
with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(RESULTS, fh, indent=1, default=float)
print("wrote results.json")
