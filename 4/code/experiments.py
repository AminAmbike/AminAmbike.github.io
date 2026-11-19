"""
SYDE 572 - Assignment 4
Runs every experiment in the report and saves the numbers to results.json
(the figures are drawn from that file by make_figures.py).

    python3 experiments.py      # about two minutes
    python3 make_figures.py

Everything except the section marked "scikit-learn" uses only the code in nb.py.
"""
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np

from nb import (read_documents, split_80_20, train_nb, score_doc_label, classify_nb,
                classify_documents, accuracy, posterior, word_contributions, wilson_interval,
                normal_interval, mcnemar_exact, cross_validate_alpha, informative_words,
                print_top_nb, mark_negation, add_bigrams, train_gaussian_nb, classify_gaussian)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "all_sentiment_shuffled.txt")
IRIS = os.path.join(HERE, "bezdekIris.data")
R = {}
t0 = time.time()


def summary(true, guess):
    k = sum(t == g for t, g in zip(true, guess))
    n = len(true)
    return {"correct": k, "n": n, "accuracy": k / n,
            "wilson": wilson_interval(k, n), "normal": normal_interval(k, n)}


# ============================================================ data and split ====
all_docs, all_labels, all_topics = read_documents(DATA, with_topics=True)
train_docs, eval_docs = split_80_20(all_docs)
train_labels, eval_labels = split_80_20(all_labels)
train_topics, eval_topics = split_80_20(all_topics)
R["data"] = {"n_docs": len(all_docs), "n_train": len(train_docs), "n_eval": len(eval_docs),
             "train_label_counts": dict(Counter(train_labels)), "eval_label_counts": dict(Counter(eval_labels)),
             "topic_label_counts": {t: dict(Counter(l for l, tt in zip(all_labels, all_topics) if tt == t))
                                    for t in sorted(set(all_topics))},
             "tokens_total": sum(len(d) for d in all_docs),
             "median_len": float(np.median([len(d) for d in all_docs])),
             "mean_len": float(np.mean([len(d) for d in all_docs]))}

# ============================================================ baseline (alpha = 1) ====
model = train_nb(train_docs, train_labels, alpha=1.0)
R["vocab_size"] = model["V"]
R["priors"] = {c: math.exp(model["log_prior"][c]) for c in model["classes"]}
sanity = {}
for doc in (["great"], ["bad"], ["excellent"], ["waste"], ["a", "top-quality", "performance"],
            ["not", "bad"], ["the", "movie"]):
    key = " ".join(doc)
    sanity[key] = {"p": {c: math.exp(score_doc_label(doc, c, model)) for c in model["classes"]},
                   "logp": {c: score_doc_label(doc, c, model) for c in model["classes"]},
                   "guess": classify_nb(doc, model), "posterior": posterior(doc, model)}
R["sanity"] = sanity
base_guess = classify_documents(eval_docs, model)
R["baseline"] = summary(eval_labels, base_guess)
R["baseline"]["confusion"] = {t: {g: sum(1 for a, b in zip(eval_labels, base_guess) if a == t and b == g)
                                   for g in model["classes"]} for t in model["classes"]}
R["baseline"]["by_topic"] = {}
for tp in sorted(set(eval_topics)):
    idx = [i for i, x in enumerate(eval_topics) if x == tp]
    R["baseline"]["by_topic"][tp] = summary([eval_labels[i] for i in idx], [base_guess[i] for i in idx])
print("baseline", R["baseline"]["accuracy"], round(time.time() - t0, 1), "s")

# ============================================================ smoothing ====
alphas = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]
cv = cross_validate_alpha(train_docs, train_labels, alphas, k=5)
test_acc = {}
for a in alphas:
    m = train_nb(train_docs, train_labels, alpha=a)
    test_acc[a] = accuracy(eval_labels, classify_documents(eval_docs, m))
best_alpha = max(alphas, key=lambda a: cv[a][0])
best_model = train_nb(train_docs, train_labels, alpha=best_alpha)
best_guess = classify_documents(eval_docs, best_model)
R["smoothing"] = {"alphas": alphas, "cv_mean": [cv[a][0] for a in alphas], "cv_sd": [cv[a][1] for a in alphas],
                  "test": [test_acc[a] for a in alphas], "best_alpha_cv": best_alpha,
                  "best": summary(eval_labels, best_guess),
                  "mcnemar_vs_alpha1": mcnemar_exact(eval_labels, best_guess, base_guess)}
print("smoothing done, best alpha", best_alpha, round(time.time() - t0, 1), "s")

# ============================================================ error analysis ====
errors = []
for i, (d, t, g) in enumerate(zip(eval_docs, eval_labels, base_guess)):
    post = posterior(d, model)
    if g != t:
        errors.append({"i": i, "true": t, "guess": g, "p_wrong": float(post[g]), "len": len(d),
                       "topic": eval_topics[i]})
R["errors"] = {"n": len(errors), "by_true": dict(Counter(e["true"] for e in errors))}
conf = [max(posterior(d, model).values()) for d in eval_docs]
correct_flags = [g == t for g, t in zip(base_guess, eval_labels)]
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.9999, 1.0000001]
calib = []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = [c for c, p in zip(correct_flags, conf) if lo <= p < hi]
    calib.append({"lo": lo, "hi": min(hi, 1.0), "n": len(sel), "acc": (sum(sel) / len(sel)) if sel else None})
R["calibration"] = calib
R["confidence"] = {"correct": [c for c, f in zip(conf, correct_flags) if f], "wrong": [c for c, f in zip(conf, correct_flags) if not f]}
lens = np.array([len(d) for d in eval_docs])
qs = np.quantile(lens, [0, 0.25, 0.5, 0.75, 1.0])
R["acc_by_length"] = []
for lo, hi in zip(qs[:-1], qs[1:]):
    sel = [f for f, L in zip(correct_flags, lens) if lo <= L <= hi]
    R["acc_by_length"].append({"lo": int(lo), "hi": int(hi), "n": len(sel), "acc": sum(sel) / len(sel)})

short = sorted([e for e in errors if e["len"] <= 30], key=lambda e: -e["p_wrong"])
long_ = sorted([e for e in errors if e["len"] > 30], key=lambda e: -e["p_wrong"])
picked = short[:8] + long_[:2]
R["error_examples"] = []
for e in picked:
    d = eval_docs[e["i"]]
    R["error_examples"].append({**e, "doc": d, "contrib": word_contributions(d, model, "pos", "neg"),
                                "logodds_prior": model["log_prior"]["pos"] - model["log_prior"]["neg"]})

# ============================================================ topic classifier ====
topic_model = train_nb(train_docs, train_topics, alpha=1.0)
topic_guess = classify_documents(eval_docs, topic_model)
R["topic"] = summary(eval_topics, topic_guess)
R["topic"]["classes"] = topic_model["classes"]
R["topic"]["confusion"] = [[sum(1 for a, b in zip(eval_topics, topic_guess) if a == t and b == g)
                            for g in topic_model["classes"]] for t in topic_model["classes"]]
print("topic", R["topic"]["accuracy"], round(time.time() - t0, 1), "s")

# ============================================================ learning curve ====
fracs = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rng = random.Random(572)
lc = {"fracs": fracs, "sent_mean": [], "sent_sd": [], "topic_mean": [], "topic_sd": [], "n_train": []}
idx_all = list(range(len(train_docs)))
for f in fracs:
    n = int(round(f * len(train_docs)))
    reps = 1 if f == 1.0 else 5
    s_acc, t_acc = [], []
    for _ in range(reps):
        idx = rng.sample(idx_all, n) if f < 1.0 else idx_all
        d = [train_docs[i] for i in idx]
        m_s = train_nb(d, [train_labels[i] for i in idx], alpha=1.0)
        m_t = train_nb(d, [train_topics[i] for i in idx], alpha=1.0)
        s_acc.append(accuracy(eval_labels, classify_documents(eval_docs, m_s)))
        t_acc.append(accuracy(eval_topics, classify_documents(eval_docs, m_t)))
    lc["n_train"].append(n)
    lc["sent_mean"].append(float(np.mean(s_acc)))
    lc["sent_sd"].append(float(np.std(s_acc, ddof=1)) if reps > 1 else 0.0)
    lc["topic_mean"].append(float(np.mean(t_acc)))
    lc["topic_sd"].append(float(np.std(t_acc, ddof=1)) if reps > 1 else 0.0)
R["learning_curve"] = lc
print("learning curve", round(time.time() - t0, 1), "s")

# ============================================================ informative words ====
info_model = train_nb(train_docs, train_labels, alpha=50.0)
ranked = informative_words(info_model, "pos", "neg", min_count=20)
top_pos, top_neg = print_top_nb(25, info_model, "pos", "neg", min_count=20)
R["informative"] = {"alpha": 50.0, "min_count": 20, "top_pos": top_pos, "top_neg": top_neg,
                    "all": [(w, s, n) for w, s, n in ranked]}
raw_ranked = informative_words(model, "pos", "neg", min_count=1)
R["informative_alpha1_raw"] = {"top_pos": raw_ranked[:12], "top_neg": raw_ranked[::-1][:12]}
# words that characterise each topic: log P(w | topic) - log P(w | all other topics)
topic_words = {}
for tp in topic_model["classes"]:
    rest = [d for d, t in zip(train_docs, train_topics) if t != tp]
    inside = [d for d, t in zip(train_docs, train_topics) if t == tp]
    m2 = train_nb(inside + rest, ["in"] * len(inside) + ["out"] * len(rest), alpha=50.0)
    topic_words[tp] = [(w, s, n) for w, s, n in informative_words(m2, "in", "out", min_count=20)[:80]]
R["topic_words"] = topic_words

# ============================================================ feature variants ====
variants = {"unigrams": lambda d: d, "negation marking": mark_negation,
            "unigrams + bigrams": add_bigrams, "negation + bigrams": lambda d: add_bigrams(mark_negation(d))}
R["features"] = {}
var_guess = {}
for name, fn in variants.items():
    trd = [fn(d) for d in train_docs]
    evd = [fn(d) for d in eval_docs]
    cvv = cross_validate_alpha(trd, train_labels, [0.1, 0.25, 0.5, 1, 2, 3, 5], k=5)
    a_best = max(cvv, key=lambda a: cvv[a][0])
    out = {}
    for a in (1.0, a_best):
        m = train_nb(trd, train_labels, alpha=a)
        g = classify_documents(evd, m)
        out[str(a)] = summary(eval_labels, g)
        if a == a_best:
            var_guess[name] = g
        out[str(a)]["vocab"] = m["V"]
    R["features"][name] = {"alpha_cv": a_best, "cv": {str(k): v for k, v in cvv.items()}, "results": out,
                           "mcnemar_vs_unigrams_alpha1": mcnemar_exact(eval_labels, var_guess[name], base_guess)}
    print("variant", name, a_best, out[str(a_best)]["accuracy"], round(time.time() - t0, 1), "s")
R["negation_example"] = " ".join(mark_negation("i did n't like the plot , but the acting is not bad at all".split()))

# ============================================================ domain sensitivity ====
topics = sorted(set(all_topics))
per_topic = {}
for tp in topics:
    d = [x for x, t in zip(all_docs, all_topics) if t == tp]
    l = [x for x, t in zip(all_labels, all_topics) if t == tp]
    (dtr, dte), (ltr, lte) = split_80_20(d), split_80_20(l)
    per_topic[tp] = (dtr, ltr, dte, lte)
dom = {"topics": topics, "matrix": [], "n_test": [len(per_topic[t][2]) for t in topics]}
for src in topics:
    m = train_nb(per_topic[src][0], per_topic[src][1], alpha=1.0)
    dom["matrix"].append([accuracy(per_topic[tgt][3], classify_documents(per_topic[tgt][2], m)) for tgt in topics])
# camera vs books in detail, with intervals and a test
cam, bk = per_topic["camera"], per_topic["books"]
m_cam = train_nb(cam[0], cam[1], alpha=1.0)
m_bk = train_nb(bk[0], bk[1], alpha=1.0)
g_cam_on_bk = classify_documents(bk[2], m_cam)
g_bk_on_bk = classify_documents(bk[2], m_bk)
g_bk_on_cam = classify_documents(cam[2], m_bk)
g_cam_on_cam = classify_documents(cam[2], m_cam)
dom["camera_books"] = {"cam_on_cam": summary(cam[3], g_cam_on_cam), "cam_on_books": summary(bk[3], g_cam_on_bk),
                       "books_on_books": summary(bk[3], g_bk_on_bk), "books_on_cam": summary(cam[3], g_bk_on_cam),
                       "mcnemar_books_test": mcnemar_exact(bk[3], g_bk_on_bk, g_cam_on_bk),
                       "mcnemar_camera_test": mcnemar_exact(cam[3], g_cam_on_cam, g_bk_on_cam),
                       "n_train_cam": len(cam[0]), "n_train_books": len(bk[0])}
# all-domain model evaluated on each topic's test set (same documents)
m_all = train_nb([x for t in topics for x in per_topic[t][0]], [x for t in topics for x in per_topic[t][1]], alpha=1.0)
dom["all_row"] = [accuracy(per_topic[t][3], classify_documents(per_topic[t][2], m_all)) for t in topics]
R["domain"] = dom
print("domain", round(time.time() - t0, 1), "s")

# ============================================================ Gaussian NB (Iris) ====
X, Y = [], []
with open(IRIS) as fh:
    for line in fh:
        p = line.strip().split(",")
        if len(p) == 5:
            X.append([float(v) for v in p[:4]])
            Y.append(p[4].replace("Iris-", ""))
X, Y = np.array(X), np.array(Y)
np.random.seed(572)
n_test = 40
eval_part = np.array([False] * len(Y))
for i in np.random.choice(range(len(Y)), replace=False, size=n_test):
    eval_part[i] = True
gm = train_gaussian_nb(X[~eval_part], Y[~eval_part])
g_iris = classify_gaussian(X[eval_part], gm)
R["iris"] = {"split": summary(list(Y[eval_part]), g_iris),
             "params": {c: {"mean": gm["params"][c]["mean"].round(4).tolist(), "sd": np.sqrt(gm["params"][c]["var"]).round(4).tolist(),
                            "prior": math.exp(gm["params"][c]["log_prior"])} for c in gm["classes"]},
             "confusion": {t: {g: int(sum(1 for a, b in zip(Y[eval_part], g_iris) if a == t and b == g)) for g in gm["classes"]}
                           for t in gm["classes"]}}
rep = []
rs = np.random.RandomState(0)
for _ in range(1000):
    ev = np.zeros(len(Y), bool)
    ev[rs.choice(len(Y), n_test, replace=False)] = True
    m = train_gaussian_nb(X[~ev], Y[~ev])
    rep.append(accuracy(list(Y[ev]), classify_gaussian(X[ev], m)))
R["iris"]["repeated"] = {"mean": float(np.mean(rep)), "sd": float(np.std(rep, ddof=1)),
                         "q025": float(np.quantile(rep, 0.025)), "q975": float(np.quantile(rep, 0.975)), "runs": 1000}

# ============================================================ scikit-learn ====
# (the only part that uses a machine learning toolkit, as the handout allows)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer


def train_sklearn(docs, labels, clf, k="all"):
    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None, lowercase=False)
    sel = SelectKBest(k=k)
    pipeline = make_pipeline(vec, sel, clf)
    pipeline.fit(docs, labels)
    return pipeline


def classify_documents_sklearn(docs, model):
    return model.predict(docs)


sk = {}
t1 = time.time()
svc = train_sklearn(train_docs, train_labels, LinearSVC())
g_svc = list(classify_documents_sklearn(eval_docs, svc))
sk["LinearSVC (TF-IDF)"] = {**summary(eval_labels, g_svc), "seconds": time.time() - t1,
                            "mcnemar_vs_nb": mcnemar_exact(eval_labels, g_svc, base_guess)}
t1 = time.time()
mlp = train_sklearn(train_docs, train_labels, MLPClassifier(random_state=572, max_iter=200, early_stopping=True), k=1000)
g_mlp = list(classify_documents_sklearn(eval_docs, mlp))
sk["MLP (TF-IDF, 1000 features)"] = {**summary(eval_labels, g_mlp), "seconds": time.time() - t1,
                                     "mcnemar_vs_nb": mcnemar_exact(eval_labels, g_mlp, base_guess)}
# scikit-learn's own multinomial NB on raw counts, as a check of nb.py
cv_ = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None, lowercase=False)
Xtr = cv_.fit_transform(train_docs)
Xev = cv_.transform(eval_docs)
mnb = MultinomialNB(alpha=1.0).fit(Xtr, train_labels)
g_mnb = list(mnb.predict(Xev))
sk["MultinomialNB (counts, alpha = 1)"] = {**summary(eval_labels, g_mnb),
                                           "agreement_with_own_nb": float(np.mean([a == b for a, b in zip(g_mnb, base_guess)]))}
gnb = GaussianNB().fit(X[~eval_part], Y[~eval_part])
sk["GaussianNB check (Iris)"] = {"agreement_with_own": float(np.mean(gnb.predict(X[eval_part]) == np.array(g_iris))),
                                 "max_mean_diff": float(max(np.abs(gnb.theta_[i] - gm["params"][c]["mean"]).max() for i, c in enumerate(gnb.classes_))),
                                 "max_var_diff": float(max(np.abs(gnb.var_[i] - gm["params"][c]["var"]).max() for i, c in enumerate(gnb.classes_)))}
R["sklearn"] = sk
print("sklearn", {k: v.get("accuracy") for k, v in sk.items()}, round(time.time() - t0, 1), "s")

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=float)
print("wrote results.json in", round(time.time() - t0, 1), "s")
