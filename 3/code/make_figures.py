"""
SYDE 572 - Assignment 3
Runs every computation, saves every figure on the project page to ../media/
and writes results.json with all the numbers quoted in the write-up.

    python3 make_figures.py
"""
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

from pca_iris import (FEATURES, SPECIES, pca_by_hand, load_iris_uci, marginal_separability,
                      loo_knn_accuracy, fisher_ratio, pca_svd, pca_sklearn, variance_explained,
                      knn_graph_components, isomap_embedding, isomap_smacof, compare_embeddings)
from sklearn.neighbors import kneighbors_graph

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "media")
os.makedirs(MEDIA, exist_ok=True)

# ---------------------------------------------------------------- style ----
INK = "#1d2330"
MUTED = "#6b6f7a"
GRID = "#e6e4df"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
AQUA = "#1baf7a"
RED = "#e34948"
LIGHT = "#c9c6bf"
SP_COL = [AQUA, BLUE, ORANGE]          # setosa, versicolor, virginica

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


def species_legend(ax, loc="best", **kw):
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=7, label=s) for s, c in zip(SPECIES, SP_COL)]
    ax.legend(handles=handles, loc=loc, **kw)


def scatter_species(ax, A, B, y, s=22, alpha=0.85):
    for c in range(3):
        ax.scatter(A[y == c], B[y == c], s=s, color=SP_COL[c], alpha=alpha, edgecolor="white", lw=0.5)


def style3d(ax):
    ax.xaxis.pane.set_facecolor(SURFACE)
    ax.yaxis.pane.set_facecolor(SURFACE)
    ax.zaxis.pane.set_facecolor(SURFACE)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor(GRID)
        a._axinfo["grid"]["color"] = GRID
    ax.tick_params(colors=MUTED, labelsize=8.5)


R = {}

# ================================================================ Part 1 ====
h = pca_by_hand()
R["p1"] = {"mean": h["mean"].tolist(), "S": h["S"].tolist(), "eigenvalues": h["eigenvalues"].tolist(),
           "w1": h["w1"].tolist(),
           "queries": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in q.items()} for q in h["queries"]]}
X5 = np.array([[-2, -1], [-1, 0], [0, 1], [1, 2], [2, 3]], dtype=float)
mu, w = h["mean"], h["w1"]

fig, ax = plt.subplots(figsize=(6.6, 5.6))
t = np.linspace(-4.2, 6.2, 2)
ax.plot(mu[0] + t * w[0], mu[1] + t * w[1], color=VIOLET, lw=2, label="first PC through the mean")
ax.annotate("", xy=mu + 1.6 * w, xytext=mu, arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=2.4))
ax.text(*(mu + 1.6 * w + np.array([-3.3, 0.35])), r"$\mathbf{w}_1=(\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}})$",
        color=VIOLET, fontsize=10)
ax.scatter(X5[:, 0], X5[:, 1], s=60, color=BLUE, zorder=4, label="data points", edgecolor="white")
ax.scatter(*mu, s=110, marker="X", color=INK, zorder=5, label=r"mean $\boldsymbol{\mu}=(0,1)$")
q1, q2 = h["queries"]
ax.scatter(*q1["point"], s=80, color=ORANGE, zorder=6, edgecolor="white")
ax.scatter(*q1["recon"], s=60, facecolor="white", edgecolor=ORANGE, lw=2, zorder=6)
ax.plot([q1["point"][0], q1["recon"][0]], [q1["point"][1], q1["recon"][1]], color=ORANGE, ls="--", lw=1.8)
ax.annotate("(3, 3)", q1["point"], textcoords="offset points", xytext=(10, -4), color=ORANGE, fontsize=10, weight="semibold")
ax.annotate("reconstruction (2.5, 3.5)\nerror = √0.5 ≈ 0.707", q1["recon"], textcoords="offset points",
            xytext=(14, -62), color=ORANGE, fontsize=9.3)
ax.scatter(*q2["point"], s=90, color=RED, zorder=7, edgecolor="white", marker="D")
ax.annotate("(3, 4) lies on the PC line\nerror = 0", q2["point"], textcoords="offset points", xytext=(12, 4),
            color=RED, fontsize=9.3, weight="semibold")
ax.set_aspect("equal")
ax.set_xlim(-3.5, 6.3)
ax.set_ylim(-2.5, 5.5)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Part 1: first principal component and reconstructions")
ax.legend(loc="lower right", fontsize=9)
save(fig, "p1_by_hand.png")

# ================================================================ Part 2 ====
X, y = load_iris_uci()
R["n"] = int(len(X))
R["class_counts"] = np.bincount(y).tolist()
R["feature_mean"] = dict(zip(FEATURES, X.mean(0).round(4).tolist()))
R["feature_var"] = dict(zip(FEATURES, X.var(0, ddof=1).round(4).tolist()))
R["class_means"] = {SPECIES[c]: X[y == c].mean(0).round(3).tolist() for c in range(3)}
R["class_ranges"] = {SPECIES[c]: {"min": X[y == c].min(0).tolist(), "max": X[y == c].max(0).tolist()} for c in range(3)}
sep = marginal_separability(X, y)
R["p2_separability"] = sep
acc = {tuple(r["idx"]): r for r in sep}

# 1 feature
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw={"wspace": 0.25})
for j, ax in enumerate(axes):
    bins = np.linspace(X[:, j].min() - 0.05, X[:, j].max() + 0.05, 26)
    for c in range(3):
        ax.hist(X[y == c, j], bins=bins, color=SP_COL[c], alpha=0.55, label=SPECIES[c])
    ax.set_xlabel(f"{FEATURES[j]} (cm)")
    a = acc[(j,)]
    ax.set_title(f"{FEATURES[j]}\n5-NN acc. {a['loo_knn'] * 100:.1f}%  ·  J = {a['fisher']:.2f}", fontsize=11)
    if j == 0:
        ax.set_ylabel("number of flowers")
axes[0].legend(loc="upper right", fontsize=8.5)
save(fig, "p2_1d.png")

# 2 features
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), gridspec_kw={"wspace": 0.25, "hspace": 0.38})
for (i, j), ax in zip(pairs, axes.ravel()):
    scatter_species(ax, X[:, i], X[:, j], y)
    a = acc[(i, j)]
    ax.set_xlabel(f"{FEATURES[i]} (cm)")
    ax.set_ylabel(f"{FEATURES[j]} (cm)")
    ax.set_title(f"5-NN acc. {a['loo_knn'] * 100:.1f}%  ·  J = {a['fisher']:.2f}", fontsize=11)
species_legend(axes[0, 0], loc="upper right", fontsize=8.5)
save(fig, "p2_2d.png")

# 3 features
triples = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
fig = plt.figure(figsize=(12.5, 10))
for n, (i, j, k) in enumerate(triples):
    ax = fig.add_subplot(2, 2, n + 1, projection="3d")
    for c in range(3):
        m = y == c
        ax.scatter(X[m, i], X[m, j], X[m, k], s=16, color=SP_COL[c], alpha=0.85, depthshade=False)
    ax.set_xlabel(FEATURES[i], fontsize=9)
    ax.set_ylabel(FEATURES[j], fontsize=9)
    ax.set_zlabel(FEATURES[k], fontsize=9)
    a = acc[(i, j, k)]
    ax.set_title(f"{FEATURES[i]} · {FEATURES[j]} · {FEATURES[k]}\n5-NN acc. {a['loo_knn'] * 100:.1f}%  ·  J = {a['fisher']:.2f}",
                 fontsize=10.5, loc="left")
    ax.view_init(elev=20, azim=-55)
    style3d(ax)
    if n == 0:
        species_legend(ax, loc="upper left", fontsize=8.5)
fig.subplots_adjust(wspace=0.05, hspace=0.12)
save(fig, "p2_3d.png")

# separability summary
srt = sorted(sep, key=lambda r: (r["fisher"]))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, gridspec_kw={"wspace": 0.06})
labels = [" + ".join(f.replace("sepal ", "S.").replace("petal ", "P.") for f in r["features"]) for r in srt]
ncol = {1: BLUE, 2: VIOLET, 3: AQUA, 4: INK}
cols = [ncol[len(r["idx"])] for r in srt]
yy = np.arange(len(srt))
axes[0].barh(yy, [r["fisher"] for r in srt], color=cols, height=0.65)
for i, r in enumerate(srt):
    axes[0].text(r["fisher"] + 0.2, i, f"{r['fisher']:.2f}", va="center", fontsize=8.5)
axes[0].set_yticks(yy)
axes[0].set_yticklabels(labels, fontsize=9)
axes[0].set_xlabel("Fisher ratio J = between-class / within-class scatter")
axes[0].set_title("Cluster separation of every marginal view")
axes[0].grid(axis="y", visible=False)
axes[1].barh(yy, [r["loo_knn"] * 100 for r in srt], color=cols, height=0.65)
for i, r in enumerate(srt):
    axes[1].text(r["loo_knn"] * 100 + 0.4, i, f"{r['loo_knn'] * 100:.1f}%", va="center", fontsize=8.5)
axes[1].set_xlim(50, 103)
axes[1].set_xlabel("leave-one-out 5-NN accuracy (%)")
axes[1].set_title("Classification accuracy")
axes[1].grid(axis="y", visible=False)
axes[1].legend(handles=[Line2D([], [], color=c, lw=8, label=f"{k} feature{'s' if k > 1 else ''}") for k, c in ncol.items()],
               loc="lower right", fontsize=9)
save(fig, "p2_separability.png")

# ================================================================ Part 3 ====
P = pca_svd(X)
skp, S_sk = pca_sklearn(X)
R["p3"] = {"components": P["components"].tolist(), "singular_values": P["singular_values"].tolist(),
           "variance": P["variance"].tolist(), "mean": P["mean"].tolist(),
           "sklearn_components": skp.components_.tolist(),
           "sklearn_explained_variance": skp.explained_variance_.tolist(),
           "max_abs_diff_vs_sklearn": float(np.abs(P["scores"] - S_sk).max()),
           "knn_pc1": loo_knn_accuracy(P["scores"][:, :1], y), "knn_pc12": loo_knn_accuracy(P["scores"][:, :2], y),
           "knn_pc123": loo_knn_accuracy(P["scores"][:, :3], y), "knn_raw": loo_knn_accuracy(X, y),
           "fisher_pc1": fisher_ratio(P["scores"][:, :1], y), "fisher_pc12": fisher_ratio(P["scores"][:, :2], y),
           "fisher_pc123": fisher_ratio(P["scores"][:, :3], y),
           "pc1_class_ranges": {SPECIES[c]: [float(P["scores"][y == c, 0].min()), float(P["scores"][y == c, 0].max())] for c in range(3)}}
Xs = (X - X.mean(0)) / X.std(0, ddof=1)
Ps = pca_svd(Xs)
R["p3_standardised"] = {"components": Ps["components"].tolist(),
                        "ratio": (Ps["singular_values"] ** 2 / np.sum(Ps["singular_values"] ** 2)).tolist(),
                        "knn_pc12": loo_knn_accuracy(Ps["scores"][:, :2], y)}
Z = P["scores"]
ev = P["variance"] / P["variance"].sum()

# PC1 only
fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
bins = np.linspace(Z[:, 0].min() - 0.1, Z[:, 0].max() + 0.1, 36)
for c in range(3):
    axes[0].hist(Z[y == c, 0], bins=bins, color=SP_COL[c], alpha=0.6, label=SPECIES[c])
axes[0].set_ylabel("number of flowers")
axes[0].set_title(f"Marginalised on PC1 only ({ev[0] * 100:.1f}% of the variance): "
                  f"5-NN acc. {R['p3']['knn_pc1'] * 100:.1f}%")
axes[0].legend(loc="upper right")
rng = np.random.default_rng(0)
for c in range(3):
    axes[1].scatter(Z[y == c, 0], rng.uniform(-0.4, 0.4, (y == c).sum()) - c, s=14, color=SP_COL[c], alpha=0.8)
axes[1].set_yticks([0, -1, -2])
axes[1].set_yticklabels(SPECIES, fontsize=9)
axes[1].set_xlabel("PC1 score (cm)")
axes[1].grid(axis="y", visible=False)
save(fig, "p3_pc1.png")

# PC1-PC2 with loadings
fig, ax = plt.subplots(figsize=(8.6, 5.6))
scatter_species(ax, Z[:, 0], Z[:, 1], y, s=26)
scale = 1.8
for j in range(4):
    v = P["components"][:2, j] * scale
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, alpha=0.75))
    off = {0: (0.08, 0.12), 1: (-0.25, 0.18), 2: (0.1, -0.22), 3: (0.12, -0.08)}[j]
    ax.text(v[0] + off[0], v[1] + off[1], FEATURES[j], fontsize=9, color=INK)
ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% of variance)")
ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
ax.set_aspect("equal")
ax.set_ylim(-1.45, 1.8)
ax.set_title(f"Marginalised on PC1 and PC2: 5-NN acc. {R['p3']['knn_pc12'] * 100:.1f}%")
species_legend(ax, loc="upper left")
ax.text(0.99, 0.02, f"arrows: feature loadings × {scale}", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
save(fig, "p3_pc2.png")

# PC1-PC3 in 3-D
fig = plt.figure(figsize=(12.5, 5.4))
for n, (el, az) in enumerate([(22, -60), (8, 20)]):
    ax = fig.add_subplot(1, 2, n + 1, projection="3d")
    for c in range(3):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=16, color=SP_COL[c], alpha=0.85, depthshade=False)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.set_zlabel("PC3", fontsize=9)
    ax.view_init(elev=el, azim=az)
    style3d(ax)
    ax.set_title(("Marginalised on PC1, PC2, PC3: " f"5-NN acc. {R['p3']['knn_pc123'] * 100:.1f}%") if n == 0 else
                 "Same points, rotated", fontsize=11, loc="left")
    if n == 0:
        species_legend(ax, loc="upper left", fontsize=8.5)
fig.subplots_adjust(wspace=0.02)
save(fig, "p3_pc3.png")

# loadings heat map
fig, ax = plt.subplots(figsize=(6.8, 3.6))
C = P["components"]
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{C[i, j]:+.3f}", ha="center", va="center", fontsize=9.5,
                color="white" if abs(C[i, j]) > 0.55 else INK)
ax.set_xticks(range(4))
ax.set_xticklabels(FEATURES, fontsize=9)
ax.set_yticks(range(4))
ax.set_yticklabels([f"PC{i + 1}  ({ev[i] * 100:.1f}%)" for i in range(4)], fontsize=9)
ax.grid(False)
ax.set_title("Principal axes (rows are unit vectors)")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
save(fig, "p3_loadings.png")

# ================================================================ Part 4 ====
V = variance_explained(X)
R["p4"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in V.items()}
Vs = variance_explained(Xs)
R["p4_standardised"] = {"ratio": Vs["ratio"].tolist(), "cumulative": Vs["cumulative"].tolist(),
                        "pcs_for_50": Vs["pcs_for_50"]}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"wspace": 0.28})
ax = axes[0]
k = np.arange(1, 5)
ax.bar(k, V["ratio"] * 100, color=BLUE, width=0.6, label="variance of each PC")
ax.plot(k, V["cumulative"] * 100, "-o", color=VIOLET, lw=2, ms=7, label="cumulative")
for i in range(4):
    ax.text(k[i], V["ratio"][i] * 100 + 2, f"{V['ratio'][i] * 100:.2f}%", ha="center", fontsize=9, color=BLUE)
    if i > 0:
        ax.text(k[i] + 0.08, V["cumulative"][i] * 100 - 6, f"{V['cumulative'][i] * 100:.2f}%", fontsize=9, color=VIOLET)
ax.axhline(50, color=RED, ls="--", lw=1.3)
ax.text(4.45, 51.5, "50%", color=RED, fontsize=9, ha="right")
ax.scatter([2], [V["cumulative"][1] * 100], s=230, facecolor="none", edgecolor=AQUA, lw=2.2, zorder=6)
ax.annotate("2-D representation\n97.77%", (2, V["cumulative"][1] * 100), textcoords="offset points",
            xytext=(-20, -62), fontsize=9.5, color=AQUA, weight="semibold")
ax.set_xticks(k)
ax.set_xticklabels([f"PC{i}" for i in k])
ax.set_ylim(0, 108)
ax.set_ylabel("% of total variance")
ax.set_title("Variance explained (scree plot)")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.3))

ax = axes[1]
kk = np.arange(0, 5)
ax.plot(kk, V["recon_error_fro2"], "-o", color=ORANGE, lw=2, ms=7)
for i in kk:
    ax.annotate(f"{V['recon_error_fro2'][i]:.2f}", (i, V["recon_error_fro2"][i]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9, color=ORANGE)
ax.set_xticks(kk)
ax.set_xlabel("number of PCs kept, k")
ax.set_ylabel(r"$\|X_c - X_c V_k V_k^\top\|_F^2$  (cm²)")
ax.set_title("Reconstruction error = sum of discarded σᵢ²")
ax.set_ylim(-40, 720)
save(fig, "p4_variance.png")

# ================================================================ Part 5 ====
ncomp, comp = knn_graph_components(X, 3)
A = comp == 0
B = ~A
Dab = cdist(X[A], X[B])
ia, ib = np.unravel_index(np.argmin(Dab), Dab.shape)
bridge = (np.where(A)[0][ia], np.where(B)[0][ib], float(Dab[ia, ib]))
R["p5_graph"] = {"components": int(ncomp), "sizes": np.bincount(comp).tolist(),
                 "species_per_component": [np.bincount(y[comp == c], minlength=3).tolist() for c in range(ncomp)],
                 "bridge": [int(bridge[0]), int(bridge[1]), bridge[2]],
                 "bridge_points": [X[bridge[0]].tolist(), X[bridge[1]].tolist()]}

iso3, Zi = isomap_embedding(X, 3)
mds, Zm = isomap_smacof(iso3)
emb = {"PCA (2 PCs)": Z[:, :2], "Isomap k=3": Zi, "Isomap k=3 + SMACOF": Zm}
for kk_ in (5, 10, 20):
    emb[f"Isomap k={kk_}"] = isomap_embedding(X, kk_)[1]
cmp_ = compare_embeddings(X, y, emb)
R["p5_compare"] = cmp_
R["p5_isomap_recon_error"] = float(iso3.reconstruction_error())
R["p5_mds_stress"] = float(mds.stress_)
G = kneighbors_graph(X, 3).toarray()
G = np.maximum(G, G.T)
geo = iso3.dist_matrix_
R["p5_geodesic"] = {"max": float(geo.max()), "euclid_max": float(cdist(X, X).max()),
                    "setosa_to_virginica_mean_geo": float(geo[np.ix_(y == 0, y == 2)].mean()),
                    "setosa_to_virginica_mean_euc": float(cdist(X[y == 0], X[y == 2]).mean())}

fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), gridspec_kw={"wspace": 0.22, "hspace": 0.3})
ax = axes[0, 0]
ii, jj = np.nonzero(np.triu(G))
for a_, b_ in zip(ii, jj):
    ax.plot(Z[[a_, b_], 0], Z[[a_, b_], 1], color=LIGHT, lw=0.8, zorder=1)
ax.plot(Z[[bridge[0], bridge[1]], 0], Z[[bridge[0], bridge[1]], 1], color=RED, lw=2.4, zorder=3)
scatter_species(ax, Z[:, 0], Z[:, 1], y, s=18)
ax.annotate("edge added to join\nthe two pieces", Z[[bridge[0], bridge[1]], :2].mean(0), textcoords="offset points",
            xytext=(25, -55), fontsize=9, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=1))
ax.set_title("3-NN graph (drawn on PC1/PC2): 2 disconnected pieces", fontsize=11.5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
species_legend(ax, loc="upper left", fontsize=8.5)

panels = [("PCA (2 PCs)", "PCA: PC1 vs PC2", ("PC1", "PC2")),
          ("Isomap k=3", "ISOMAP, 3 nearest neighbours", ("Isomap 1", "Isomap 2")),
          ("Isomap k=3 + SMACOF", "Same geodesic distances, iterative MDS", ("MDS 1", "MDS 2"))]
for ax, (key, title, (xl, yl)) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], panels):
    E = emb[key]
    scatter_species(ax, E[:, 0], E[:, 1], y, s=22)
    m = cmp_[key]
    ax.set_title(f"{title}\n5-NN acc. {m['loo_knn'] * 100:.1f}%  ·  trustworthiness {m['trustworthiness']:.3f}", fontsize=11.5)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
save(fig, "p5_isomap.png")

# metric comparison
names = list(cmp_.keys())
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
cols = [VIOLET] + [ORANGE, RED] + [BLUE] * 3
yy = np.arange(len(names))[::-1]
for ax, (key, lab, lo, hi, fmt) in zip(axes, [("loo_knn", "leave-one-out 5-NN accuracy (%)", 88, 100, "{:.1f}%"),
                                               ("trustworthiness", "trustworthiness (5 neighbours)", 0.90, 1.0, "{:.3f}"),
                                               ("resid_var_euclid", "residual variance vs. Euclidean distances", 0, 0.06, "{:.3f}")]):
    vals = [cmp_[n][key] * (100 if key == "loo_knn" else 1) for n in names]
    ax.barh(yy, vals, color=cols, height=0.62)
    for yv, v in zip(yy, vals):
        ax.text(v + (hi - lo) * 0.01, yv, fmt.format(v), va="center", fontsize=8.8)
    ax.set_xlim(lo, hi + (hi - lo) * 0.14)
    ax.set_xlabel(lab, fontsize=9.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(yy)
axes[0].set_yticklabels(names, fontsize=9.5)
axes[0].set_title("Classification")
axes[1].set_title("Neighbourhood preservation")
axes[2].set_title("Distance preservation (lower is better)")
save(fig, "p5_compare.png")

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=float)
print("wrote results.json")
