"""
SYDE 572 - Assignment 3
PCA by hand, marginal views of Iris, PCA, data compression and Isomap.

Functions
    pca_by_hand             Part 1  (the 5-point dataset)
    load_iris_uci           Parts 2 to 5
    marginal_separability   Part 2  (how well each feature subset separates the species)
    pca_svd                 Part 3  (PCA from the SVD of the centred data matrix)
    variance_explained      Part 4  (Frobenius norm and explained variance)
    isomap_embedding        Part 5  (ISOMAP with k nearest neighbours)
    compare_embeddings      Part 5  (PCA vs ISOMAP: compression and classification)
"""
import itertools
import os

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, trustworthiness
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "bezdekIris.data")
FEATURES = ["sepal length", "sepal width", "petal length", "petal width"]
SPECIES = ["setosa", "versicolor", "virginica"]


# ================================================================ Part 1 ====
def pca_by_hand(X=None, queries=((3, 3), (3, 4))):
    """
    First principal component of a small 2-D dataset and the L2
    reconstruction error of new points, following the hand calculation:
      1. centre the data with the sample mean
      2. covariance matrix S = (1/N) Xc^T Xc
      3. first PC = unit eigenvector of S with the largest eigenvalue
      4. a point p is reconstructed as  mu + ((p - mu) . w) w
    """
    if X is None:
        X = np.array([[-2, -1], [-1, 0], [0, 1], [1, 2], [2, 3]], dtype=float)
    mu = X.mean(axis=0)
    Xc = X - mu
    S = Xc.T @ Xc / len(X)
    evals, evecs = np.linalg.eigh(S)            # ascending order
    w = evecs[:, -1]
    w = w * np.sign(w[np.argmax(np.abs(w))])    # fix the sign so the largest entry is positive
    out = {"mean": mu, "S": S, "eigenvalues": evals[::-1], "w1": w, "queries": []}
    for q in queries:
        p = np.asarray(q, dtype=float)
        z = (p - mu) @ w                        # 1-D code (coordinate along the PC)
        p_hat = mu + z * w                      # reconstruction
        out["queries"].append({"point": p, "centred": p - mu, "z": z, "recon": p_hat,
                               "residual": p - p_hat, "error": float(np.linalg.norm(p - p_hat))})
    return out


# ============================================================ Iris data ====
def load_iris_uci(path=DATA_FILE):
    """
    Read the UCI Iris file (comma separated: 4 measurements in cm, species).
    bezdekIris.data is the corrected file in the UCI archive; it matches
    Fisher's published table (samples 35 and 38 are fixed).
    """
    X, y = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            X.append([float(v) for v in parts[:4]])
            y.append(SPECIES.index(parts[4].replace("Iris-", "")))
    return np.array(X), np.array(y)


# ================================================================ Part 2 ====
def loo_knn_accuracy(Z, y, k=5):
    """Leave-one-out accuracy of a k-NN classifier: a simple measure of how
    well separated the classes are in the representation Z."""
    clf = KNeighborsClassifier(n_neighbors=k)
    return float(cross_val_score(clf, Z, y, cv=LeaveOneOut()).mean())


def fisher_ratio(Z, y):
    """Between-class scatter / within-class scatter (trace ratio).
    Large values mean tight, well separated clusters."""
    mu = Z.mean(axis=0)
    sb = sum((y == c).sum() * np.sum((Z[y == c].mean(axis=0) - mu) ** 2) for c in np.unique(y))
    sw = sum(np.sum((Z[y == c] - Z[y == c].mean(axis=0)) ** 2) for c in np.unique(y))
    return float(sb / sw)


def marginal_separability(X, y, k=5):
    """Score every subset of 1 to 4 features (the marginal views)."""
    rows = []
    for r in (1, 2, 3, 4):
        for idx in itertools.combinations(range(X.shape[1]), r):
            Z = X[:, idx]
            rows.append({"features": [FEATURES[i] for i in idx], "idx": list(idx),
                         "loo_knn": loo_knn_accuracy(Z, y, k), "fisher": fisher_ratio(Z, y)})
    return rows


# ================================================================ Part 3 ====
def pca_svd(X):
    """
    PCA from the SVD of the centred data matrix  Xc = U Sigma V^T.
    Rows of V^T are the principal axes; Xc V are the scores;
    the variance along PC i is sigma_i^2 / (N - 1).
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    # same sign convention as scikit-learn (largest |loading| of each axis positive)
    signs = np.sign(Vt[np.arange(len(s)), np.argmax(np.abs(Vt), axis=1)])
    Vt = Vt * signs[:, None]
    scores = Xc @ Vt.T
    return {"mean": mu, "singular_values": s, "components": Vt, "scores": scores,
            "variance": s ** 2 / (len(X) - 1)}


def pca_sklearn(X, n_components=None):
    """The same decomposition with scikit-learn, used as a check."""
    p = PCA(n_components=n_components).fit(X)
    return p, p.transform(X)


# ================================================================ Part 4 ====
def variance_explained(X):
    """
    For a centred matrix Xc,  ||Xc||_F^2 = trace(Xc^T Xc) = sum_i sigma_i^2,
    which is (N - 1) times the total variance.  Keeping k PCs leaves the
    reconstruction error  ||Xc - Xc V_k V_k^T||_F^2 = sum_{i>k} sigma_i^2,
    so the fraction of variance explained by k PCs is
        sum_{i<=k} sigma_i^2 / ||Xc||_F^2.
    """
    Xc = X - X.mean(axis=0)
    fro2 = float(np.sum(Xc ** 2))
    s = np.linalg.svd(Xc, compute_uv=False)
    Vt = np.linalg.svd(Xc, full_matrices=False)[2]
    ratio = s ** 2 / fro2
    recon = []
    for k in range(len(s) + 1):
        Vk = Vt[:k].T
        R = Xc - Xc @ Vk @ Vk.T
        recon.append(float(np.sum(R ** 2)))
    cum = np.cumsum(ratio)
    return {"fro2": fro2, "singular_values": s, "sigma2": s ** 2, "ratio": ratio,
            "cumulative": cum, "recon_error_fro2": recon,
            "pcs_for_50": int(np.searchsorted(cum, 0.5) + 1),
            "pcs_for_95": int(np.searchsorted(cum, 0.95) + 1)}


# ================================================================ Part 5 ====
def knn_graph_components(X, k=3):
    """Number of connected pieces of the (symmetrised) k-NN graph."""
    G = kneighbors_graph(X, k, mode="distance")
    n, labels = connected_components(G, directed=False)
    return n, labels


def isomap_embedding(X, k=3, n_components=2):
    """
    ISOMAP: k-NN graph -> geodesic (shortest-path) distances -> MDS.
    scikit-learn's Isomap performs the final MDS step in closed form
    (classical MDS = eigendecomposition of the double-centred squared
    geodesic distances).  If the k-NN graph is disconnected it joins the
    pieces with their shortest Euclidean edge before computing paths.
    """
    iso = Isomap(n_neighbors=k, n_components=n_components)
    Z = iso.fit_transform(X)
    return iso, Z


def isomap_smacof(iso, n_components=2, seed=0):
    """
    The variant described in the handout: take the geodesic distance matrix
    computed by Isomap and fit metric MDS iteratively (SMACOF, an iterative
    stress-minimisation method closely related to gradient descent).
    """
    D = iso.dist_matrix_
    mds = MDS(n_components=n_components, dissimilarity="precomputed", metric=True,
              n_init=4, max_iter=1000, random_state=seed, normalized_stress="auto")
    Z = mds.fit_transform(D)
    return mds, Z


def residual_variance(D_target, Z):
    """1 - R^2 between the target distances and the distances in the embedding
    (the measure used in the original ISOMAP paper; lower is better)."""
    d_emb = pdist(Z)
    d_tgt = squareform(D_target, checks=False)
    r = np.corrcoef(d_tgt, d_emb)[0, 1]
    return float(1 - r ** 2)


def compare_embeddings(X, y, embeddings, k_eval=5):
    """
    Score 2-D embeddings on
      - LOO k-NN accuracy (can the species still be classified?)
      - Fisher ratio (how tight and separated are the clusters?)
      - trustworthiness (are the 2-D neighbours real neighbours in 4-D?)
      - residual variance against Euclidean and against geodesic distances.
    """
    D_euc = squareform(pdist(X))
    iso_geo = Isomap(n_neighbors=3, n_components=2).fit(X).dist_matrix_
    out = {}
    for name, Z in embeddings.items():
        out[name] = {"loo_knn": loo_knn_accuracy(Z, y, k_eval),
                     "fisher": fisher_ratio(Z, y),
                     "trustworthiness": float(trustworthiness(X, Z, n_neighbors=k_eval)),
                     "resid_var_euclid": residual_variance(D_euc, Z),
                     "resid_var_geodesic": residual_variance(iso_geo, Z)}
    return out


if __name__ == "__main__":
    h = pca_by_hand()
    print("first PC:", h["w1"], " eigenvalues:", h["eigenvalues"])
    for q in h["queries"]:
        print(q["point"], "-> recon", q["recon"], " error", round(q["error"], 6))
    X, y = load_iris_uci()
    v = variance_explained(X)
    print("explained variance ratio:", np.round(v["ratio"], 4), " cumulative:", np.round(v["cumulative"], 4))
    print("PCs needed for 50%:", v["pcs_for_50"])
    print("k-NN graph components (k=3):", knn_graph_components(X, 3)[0])
