"""
SYDE 572 - Assignment 4
Draws every figure in the report from results.json (run experiments.py first).

    python3 make_figures.py
"""
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "media")
os.makedirs(MEDIA, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))

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
POS, NEG = AQUA, RED
TOPIC_COL = {"books": VIOLET, "camera": BLUE, "dvd": ORANGE, "health": AQUA, "music": "#c2408f", "software": "#8a6a12"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f1ea", "#bcd6f5", "#2a78d6", "#0d3a73"])

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "mathtext.fontset": "stixsans", "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.labelcolor": INK, "axes.edgecolor": LIGHT, "axes.linewidth": 0.8, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "legend.frameon": False, "legend.fontsize": 9.5, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})


def save(fig, name):
    fig.savefig(os.path.join(MEDIA, name))
    plt.close(fig)
    print("saved", name)


def heat(ax, M, rows, cols, fmt, cmap=SEQ, vmin=None, vmax=None, hl_diag=False):
    M = np.asarray(M, dtype=float)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    hi = np.nanmax(M) if vmax is None else vmax
    lo = np.nanmin(M) if vmin is None else vmin
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9.5,
                    color="white" if (v - lo) / (hi - lo + 1e-12) > 0.6 else INK,
                    weight="semibold" if (hl_diag and i == j) else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


# ================================================================ smoothing ====
S = R["smoothing"]
a = np.array(S["alphas"])
fig, ax = plt.subplots(figsize=(8.4, 4.4))
cvm, cvs = np.array(S["cv_mean"]) * 100, np.array(S["cv_sd"]) * 100
ax.fill_between(a, cvm - cvs, cvm + cvs, color=VIOLET, alpha=0.13, lw=0, label="5-fold CV on training set, ± 1 s.d.")
ax.semilogx(a, cvm, "-o", color=VIOLET, lw=2, ms=6, label="5-fold CV accuracy (mean)")
ax.semilogx(a, np.array(S["test"]) * 100, "-o", color=ORANGE, lw=2, ms=6, label="accuracy on the 2,383 test reviews")
ib = S["alphas"].index(S["best_alpha_cv"])
ax.scatter([a[ib]], [cvm[ib]], s=220, facecolor="none", edgecolor=AQUA, lw=2.2, zorder=6)
ax.annotate(f"CV choice α = {S['best_alpha_cv']:g}", (a[ib], cvm[ib]), textcoords="offset points", xytext=(12, -32),
            color=AQUA, fontsize=9.5, weight="semibold")
ax.axvline(1, color=MUTED, ls=":", lw=1)
ax.text(1.06, 82.6, "Laplace (α = 1)", color=MUTED, fontsize=8.8)
ax.set_xlabel("smoothing parameter α (log scale)")
ax.set_ylabel("accuracy (%)")
ax.set_ylim(75.5, 83)
ax.set_title("Effect of additive smoothing")
ax.legend(loc="upper left", fontsize=9)
save(fig, "f_smoothing.png")

# ============================================================ per topic accuracy ====
B = R["baseline"]
topics = sorted(B["by_topic"])
fig, ax = plt.subplots(figsize=(8.4, 3.8))
vals = [B["by_topic"][t]["accuracy"] * 100 for t in topics] + [B["accuracy"] * 100]
lo = [B["by_topic"][t]["wilson"][0] * 100 for t in topics] + [B["wilson"][0] * 100]
hi = [B["by_topic"][t]["wilson"][1] * 100 for t in topics] + [B["wilson"][1] * 100]
names = topics + ["all"]
cols = [TOPIC_COL[t] for t in topics] + [INK]
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=cols, height=0.6, alpha=0.9)
ax.errorbar(vals, y, xerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, capsize=3, lw=1.2)
for yy, v, h in zip(y, vals, hi):
    ax.text(h + 0.4, yy, f"{v:.1f}%", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([f"{n} (n = {B['by_topic'][n]['n']})" if n != "all" else f"all (n = {B['n']})" for n in names])
ax.set_xlim(70, 95)
ax.set_xlabel("sentiment accuracy on the test set (%), with 95% Wilson intervals")
ax.set_title("Sentiment accuracy (α = 1) split by review topic")
ax.grid(axis="y", visible=False)
save(fig, "f_by_topic.png")

# ============================================================ confidence ====
C = R["confidence"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.25})
ax = axes[0]
bins = np.linspace(0.5, 1.0, 26)
ax.hist(C["correct"], bins=bins, color=BLUE, alpha=0.7, label=f"correct ({len(C['correct'])})")
ax.hist(C["wrong"], bins=bins, color=RED, alpha=0.75, label=f"wrong ({len(C['wrong'])})")
ax.set_yscale("log")
ax.set_xlabel("posterior probability of the predicted label")
ax.set_ylabel("number of test reviews (log scale)")
ax.set_title("Naive Bayes is very confident, even when wrong")
ax.legend(loc="upper left")
ax = axes[1]
cal = [c for c in R["calibration"] if c["n"]]
labels = [f"{c['lo']:.4g} to {c['hi']:.4g}".replace("1.0000001", "1") for c in cal]
x = np.arange(len(cal))
mid = [min((c["lo"] + c["hi"]) / 2, 0.99995) for c in cal]
ax.bar(x, [c["acc"] * 100 for c in cal], color=BLUE, width=0.6, label="observed accuracy")
ax.plot(x, [m * 100 for m in mid], "_", color=RED, ms=30, mew=2.5, label="accuracy if calibrated (bin centre)")
for xi, c in zip(x, cal):
    ax.text(xi, c["acc"] * 100 + 1.5, f"{c['acc'] * 100:.0f}%\nn={c['n']}", ha="center", fontsize=8.3)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=20)
ax.set_ylim(0, 125)
ax.set_xlabel("confidence bin")
ax.set_ylabel("accuracy in bin (%)")
ax.set_title("Reliability: stated confidence vs actual accuracy")
ax.legend(loc="upper left", fontsize=8.8)
save(fig, "f_confidence.png")

# ============================================================ topic classifier ====
T = R["topic"]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
M = np.array(T["confusion"], dtype=float)
im = heat(ax, M, [f"{c}" for c in T["classes"]], T["classes"], lambda v: f"{int(v)}", hl_diag=True)
ax.set_xlabel("predicted topic")
ax.set_ylabel("true topic")
ax.set_title(f"Six-topic classifier: accuracy {T['accuracy'] * 100:.1f}% on 2,383 test reviews")
save(fig, "f_topic_confusion.png")

# ============================================================ learning curve ====
L = R["learning_curve"]
n = np.array(L["n_train"])
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for key, col, lab in (("sent", ORANGE, "sentiment (2 classes)"), ("topic", BLUE, "topic (6 classes)")):
    m, s = np.array(L[key + "_mean"]) * 100, np.array(L[key + "_sd"]) * 100
    ax.fill_between(n, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.semilogx(n, m, "-o", color=col, lw=2, ms=5.5, label=lab)
    ax.annotate(f"{m[-1]:.1f}%", (n[-1], m[-1]), textcoords="offset points", xytext=(6, -3), color=col, fontsize=9.5, weight="semibold")
for f, nn in zip(L["fracs"], n):
    if f in (0.02, 0.1, 0.5, 1.0):
        ax.text(nn, 59, f"{f * 100:g}%", ha="center", fontsize=8.3, color=MUTED)
ax.set_xlabel("number of training reviews (log scale; labels show % of the 80% training split)")
ax.set_ylabel("test accuracy (%)")
ax.set_ylim(57, 95)
ax.set_title("Learning curves (α = 1; mean ± 1 s.d. over 5 random subsets)")
ax.legend(loc="upper left")
save(fig, "f_learning_curve.png")

# ============================================================ informative words ====
I = R["informative"]
k = 20
pos, neg = I["top_pos"][:k], I["top_neg"][:k]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), gridspec_kw={"wspace": 0.45})
for ax, items, col, title in ((axes[0], pos, POS, "Most positive words"), (axes[1], neg, NEG, "Most negative words")):
    words = [w for w, _, _ in items][::-1]
    sc = [abs(s) for _, s, _ in items][::-1]
    cnt = [c for _, _, c in items][::-1]
    yy = np.arange(len(words))
    ax.barh(yy, sc, color=col, height=0.65, alpha=0.9)
    for yi, s_, c_ in zip(yy, sc, cnt):
        ax.text(s_ + 0.02, yi, f"{s_:.2f}  ({c_})", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(yy)
    ax.set_yticklabels(words, fontsize=9.5)
    ax.set_xlim(0, max(sc) * 1.35)
    ax.set_xlabel("|log P(w | pos) − log P(w | neg)|   (count in brackets)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
save(fig, "f_informative.png")

# scatter: frequency vs log-ratio
allw = I["all"]
fig, ax = plt.subplots(figsize=(10, 5.4))
cnt = np.array([c for _, _, c in allw], dtype=float)
sc = np.array([s for _, s, _ in allw])
ax.scatter(cnt, sc, s=6, c=np.where(sc > 0, POS, NEG), alpha=0.25, lw=0)
show = {"excellent", "easy", "great", "best", "love", "recommend", "good", "the", "not", "waste", "worst", "boring", "horrible",
        "support", "refund", "bad", "money", "disappointed", "wonderful", "lenses", "warranty", "highly"}
for w, s, c in allw:
    if w in show:
        ax.annotate(w, (c, s), fontsize=8.6, color=POS if s > 0 else NEG, xytext=(3, 2), textcoords="offset points")
ax.set_xscale("log")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("number of times the word occurs in the training reviews (log scale)")
ax.set_ylabel("log P(w | pos) − log P(w | neg)")
ax.set_title(f"Every word seen at least {I['min_count']} times (α = {I['alpha']:g})")
save(fig, "f_word_scatter.png")

# word clouds
try:
    from wordcloud import WordCloud
    font = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
    font = font if os.path.exists(font) else None
    freq, colour = {}, {}
    for w, s, c in [t for t in I["all"] if t[1] > 0 and t[0].isalpha()][:110]:
        freq[w] = s * np.log(c)
        colour[w] = POS
    for w, s, c in [t for t in I["all"][::-1] if t[1] < 0 and t[0].isalpha()][:110]:
        freq[w] = -s * np.log(c)
        colour[w] = NEG
    wc = WordCloud(width=1300, height=650, background_color=SURFACE, font_path=font, prefer_horizontal=0.92,
                   random_state=3, max_words=220, margin=4, relative_scaling=0.35, max_font_size=140).generate_from_frequencies(freq)
    wc.recolor(color_func=lambda word, **kw: colour.get(word, INK))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    save(fig, "f_cloud_polarity.png")
    freq, colour = {}, {}
    for tp, items in R["topic_words"].items():
        for w, s, c in [t for t in items if t[0].isalpha()][:40]:
            v = s * np.log(c)
            if v > freq.get(w, 0):
                freq[w] = v
                colour[w] = TOPIC_COL[tp]
    wc = WordCloud(width=1300, height=650, background_color=SURFACE, font_path=font, prefer_horizontal=0.92,
                   random_state=5, max_words=240, margin=4, relative_scaling=0.35, max_font_size=140).generate_from_frequencies(freq)
    wc.recolor(color_func=lambda word, **kw: colour.get(word, INK))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    save(fig, "f_cloud_topic.png")
except ImportError:
    print("wordcloud not installed; skipping the word clouds")

# ============================================================ feature variants and toolkits ====
F = R["features"]
SK = R["sklearn"]
rows = [("NB, unigrams (α = 1)", R["baseline"], INK)]
for name in ("unigrams", "negation marking", "unigrams + bigrams", "negation + bigrams"):
    a_ = F[name]["alpha_cv"]
    rows.append((f"NB, {name} (CV α = {a_:g})", F[name]["results"][str(float(a_)) if str(float(a_)) in F[name]["results"] else str(a_)], VIOLET))
rows.append(("scikit-learn LinearSVC (TF-IDF)", SK["LinearSVC (TF-IDF)"], ORANGE))
rows.append(("scikit-learn MLP (TF-IDF, 1000 feat.)", SK["MLP (TF-IDF, 1000 features)"], ORANGE))
fig, ax = plt.subplots(figsize=(9.5, 4.4))
y = np.arange(len(rows))[::-1]
v = np.array([r[1]["accuracy"] * 100 for r in rows])
lo = np.array([r[1]["wilson"][0] * 100 for r in rows])
hi = np.array([r[1]["wilson"][1] * 100 for r in rows])
ax.barh(y, v, color=[r[2] for r in rows], height=0.6, alpha=0.88)
ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor=INK, capsize=3, lw=1.2)
for yy, vv, hh in zip(y, v, hi):
    ax.text(hh + 0.25, yy, f"{vv:.2f}%", va="center", fontsize=9)
ax.axvline(R["baseline"]["accuracy"] * 100, color=MUTED, ls=":", lw=1)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.set_xlim(78, 89.5)
ax.set_xlabel("test accuracy (%), 95% Wilson intervals")
ax.set_title("Feature variants and scikit-learn classifiers")
ax.grid(axis="y", visible=False)
save(fig, "f_features.png")

# ============================================================ domain ====
D = R["domain"]
M = np.array(D["matrix"] + [D["all_row"]]) * 100
fig, ax = plt.subplots(figsize=(8.2, 5.6))
heat(ax, M, [f"train: {t}" for t in D["topics"]] + ["train: all six"], [f"test: {t}" for t in D["topics"]],
     lambda v: f"{v:.1f}", vmin=55, vmax=90, hl_diag=True)
ax.axhline(len(D["topics"]) - 0.5, color="white", lw=4)
ax.set_title("Sentiment accuracy (%) across domains (α = 1)")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
save(fig, "f_domain.png")

# ============================================================ Gaussian NB on Iris ====
IR = R["iris"]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"wspace": 0.28})
feat = ["sepal length", "sepal width", "petal length", "petal width"]
sp_col = {"setosa": AQUA, "versicolor": BLUE, "virginica": ORANGE}
for ax, j in zip(axes[:2], (2, 3)):
    xs = np.linspace(0, 7.5 if j == 2 else 2.8, 400)
    for c, p in IR["params"].items():
        mu, sd = p["mean"][j], p["sd"][j]
        ax.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), color=sp_col[c], lw=2, label=f"{c}: μ={mu:.2f}, σ={sd:.2f}")
        ax.fill_between(xs, 0, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), color=sp_col[c], alpha=0.12)
    ax.set_xlabel(f"{feat[j]} (cm)")
    ax.set_ylabel("class-conditional density")
    ax.set_title(f"Fitted Gaussians: {feat[j]}")
    ax.legend(fontsize=8.3, loc="upper right")
ax = axes[2]
rep = IR["repeated"]
ax.bar([0], [IR["split"]["accuracy"] * 100], color=VIOLET, width=0.5)
ax.errorbar([1], [rep["mean"] * 100], yerr=[[rep["mean"] * 100 - rep["q025"] * 100], [rep["q975"] * 100 - rep["mean"] * 100]],
            fmt="o", color=ORANGE, capsize=5, ms=8, lw=2)
ax.text(0, IR["split"]["accuracy"] * 100 + 1, f"{IR['split']['accuracy'] * 100:.1f}%\n({IR['split']['correct']}/40)", ha="center", fontsize=9)
ax.text(1.12, rep["mean"] * 100, f"mean {rep['mean'] * 100:.1f}%\n95% of runs:\n{rep['q025'] * 100:.0f}% to {rep['q975'] * 100:.0f}%", fontsize=9, va="center")
ax.set_xticks([0, 1])
ax.set_xticklabels(["the split in the\nhandout code", "1,000 random\n110 / 40 splits"])
ax.set_xlim(-0.6, 1.9)
ax.set_ylim(80, 103)
ax.set_ylabel("test accuracy (%)")
ax.set_title("Gaussian NB accuracy on Iris")
save(fig, "f_iris_gnb.png")
print("done")
