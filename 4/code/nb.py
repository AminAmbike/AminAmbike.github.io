"""
SYDE 572 - Assignment 4
Multinomial Naive Bayes for text, written from scratch (no machine learning toolkit),
plus the statistics used to evaluate it and a Gaussian Naive Bayes for numerical data.

Model
    P(c | d)  proportional to  P(c) * prod_{w in d} P(w | c)

    log P(d, c) = log P(c) + sum_{w in d} log P(w | c)

Estimates (with additive / Laplace smoothing alpha)
    P(c)      = N_c / N                                   (fraction of training documents in class c)
    P(w | c)  = (count(w, c) + alpha) / (T_c + alpha (V + 1))

where count(w, c) is how often word w occurs in class-c training documents, T_c is the total
number of tokens in class c and V is the training vocabulary size.  The "+ 1" reserves one
extra slot for words never seen in training, so an unseen word gets alpha / (T_c + alpha (V + 1))
and the probabilities over the V known words plus "unknown" still sum to one.
"""
import math
from codecs import open
from collections import Counter

import numpy as np

NEGATORS = {"not", "no", "never", "n't", "cannot", "nothing", "nobody", "none", "nor", "neither", "without"}
CLAUSE_END = {".", ",", "!", "?", ";", ":", "(", ")", "--", "but"}


# ================================================================== data ====
def read_documents(doc_file, label_column=1, with_topics=False):
    """
    One review per line:  topic  polarity  id  word word word ...
    label_column = 1 gives the sentiment label, 0 gives the topic label.
    """
    docs, labels, topics = [], [], []
    with open(doc_file, encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 3:
                continue
            docs.append(words[3:])
            labels.append(words[label_column])
            topics.append(words[0])
    if with_topics:
        return docs, labels, topics
    return docs, labels


def split_80_20(items, fraction=0.80):
    split_point = int(fraction * len(items))
    return items[:split_point], items[split_point:]


# ============================================================ train / score ====
def train_nb(documents, labels, alpha=1.0):
    """
    Estimate the Naive Bayes parameters.  Returns a dict with
        classes, log_prior[c], log_likelihood[c][w], log_unknown[c], plus the raw counts
    so that other quantities (e.g. informative words) can be computed later.
    The classes are read from the labels, so any number of categories works.
    """
    classes = sorted(set(labels))
    doc_counts = Counter(labels)
    word_counts = {c: Counter() for c in classes}
    for doc, c in zip(documents, labels):
        word_counts[c].update(doc)

    vocab = set()
    for c in classes:
        vocab.update(word_counts[c])
    V = len(vocab)
    n_docs = len(documents)

    log_prior, log_likelihood, log_unknown, totals = {}, {}, {}, {}
    for c in classes:
        log_prior[c] = math.log(doc_counts[c] / n_docs)
        total = sum(word_counts[c].values())
        totals[c] = total
        denom = total + alpha * (V + 1)
        log_unknown[c] = math.log(alpha / denom)
        log_likelihood[c] = {w: math.log((word_counts[c][w] + alpha) / denom) for w in vocab}

    return {"classes": classes, "alpha": alpha, "V": V, "log_prior": log_prior,
            "log_likelihood": log_likelihood, "log_unknown": log_unknown,
            "word_counts": word_counts, "doc_counts": doc_counts, "totals": totals}


def score_doc_label(document, label, model):
    """log P(document, label) = log P(label) + sum over tokens of log P(token | label)."""
    ll = model["log_likelihood"][label]
    unk = model["log_unknown"][label]
    return model["log_prior"][label] + sum(ll.get(w, unk) for w in document)


def classify_nb(document, model):
    """The label with the highest joint log probability."""
    return max(model["classes"], key=lambda c: score_doc_label(document, c, model))


def classify_documents(docs, model):
    return [classify_nb(d, model) for d in docs]


def accuracy(true_labels, guessed_labels):
    correct = sum(t == g for t, g in zip(true_labels, guessed_labels))
    return correct / len(true_labels)


def posterior(document, model):
    """P(c | document) for every class, computed stably with log-sum-exp."""
    s = np.array([score_doc_label(document, c, model) for c in model["classes"]])
    s = s - s.max()
    p = np.exp(s)
    return dict(zip(model["classes"], p / p.sum()))


def word_contributions(document, model, a="pos", b="neg"):
    """log P(w | a) - log P(w | b) for every token: how much each word pushes towards a."""
    la, lb = model["log_likelihood"][a], model["log_likelihood"][b]
    ua, ub = model["log_unknown"][a], model["log_unknown"][b]
    return [(w, la.get(w, ua) - lb.get(w, ub)) for w in document]


# ============================================================ statistics ====
def wilson_interval(k, n, z=1.959964):
    """95% Wilson score interval for a binomial proportion k / n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def normal_interval(k, n, z=1.959964):
    """95% normal-approximation (Wald) interval, p +/- z sqrt(p(1-p)/n)."""
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


def mcnemar_exact(true_labels, guesses_a, guesses_b):
    """
    Exact McNemar test for two classifiers evaluated on the same documents.
    b = documents A gets right and B gets wrong, c = the reverse.  Under H0 (equal accuracy)
    b ~ Binomial(b + c, 1/2).  Returns b, c and the two-sided p-value.
    """
    b = sum(ga == t and gb != t for t, ga, gb in zip(true_labels, guesses_a, guesses_b))
    c = sum(ga != t and gb == t for t, ga, gb in zip(true_labels, guesses_a, guesses_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def cross_validate_alpha(docs, labels, alphas, k=5):
    """k-fold cross-validation accuracy on the training set for each smoothing value."""
    n = len(docs)
    edges = np.linspace(0, n, k + 1).astype(int)
    res = {a: [] for a in alphas}
    for f in range(k):
        lo, hi = edges[f], edges[f + 1]
        trd, trl = docs[:lo] + docs[hi:], labels[:lo] + labels[hi:]
        vad, val = docs[lo:hi], labels[lo:hi]
        for a in alphas:
            m = train_nb(trd, trl, alpha=a)
            res[a].append(accuracy(val, classify_documents(vad, m)))
    return {a: (float(np.mean(v)), float(np.std(v, ddof=1))) for a, v in res.items()}


# ======================================================== informative words ====
def informative_words(model, a="pos", b="neg", min_count=10):
    """
    Rank words by the log-likelihood ratio  log P(w | a) - log P(w | b).
    Large positive values are strong evidence for a, large negative for b.
    Rare words get extreme ratios by chance, so words seen fewer than
    min_count times in total are skipped (a large alpha has a similar effect).
    """
    la, lb = model["log_likelihood"][a], model["log_likelihood"][b]
    out = []
    for w in la:
        n = model["word_counts"][a][w] + model["word_counts"][b][w]
        if n >= min_count:
            out.append((w, la[w] - lb[w], n))
    return sorted(out, key=lambda t: t[1], reverse=True)


def print_top_nb(k, model, a="pos", b="neg", min_count=10):
    ranked = informative_words(model, a, b, min_count)
    print(f"top {k} words for {a}:", ", ".join(w for w, _, _ in ranked[:k]))
    print(f"top {k} words for {b}:", ", ".join(w for w, _, _ in ranked[::-1][:k]))
    return ranked[:k], ranked[::-1][:k]


# ========================================================= feature variants ====
def mark_negation(doc):
    """
    Pang et al. (2002) trick: after a negation word, prefix every token with NOT_
    until the next punctuation mark (or 'but'), so 'not good' becomes 'not NOT_good'.
    """
    out, negating = [], False
    for w in doc:
        if w in CLAUSE_END:
            negating = False
            out.append(w)
        elif negating:
            out.append("NOT_" + w)
        else:
            out.append(w)
        if w in NEGATORS or w.endswith("n't"):
            negating = True
    return out


def add_bigrams(doc):
    """Unigrams plus every adjacent word pair, written 'w1__w2'."""
    return doc + [doc[i] + "__" + doc[i + 1] for i in range(len(doc) - 1)]


# =================================================== Gaussian Naive Bayes ====
def train_gaussian_nb(X, y):
    """
    Numerical Naive Bayes: one Gaussian per (class, feature) with the maximum-likelihood
    mean and variance (dividing by N, not N - 1), plus the class priors.
    """
    X = np.asarray(X, dtype=float)
    classes = sorted(set(y))
    y = np.asarray(y)
    params = {}
    for c in classes:
        Xc = X[y == c]
        params[c] = {"log_prior": math.log(len(Xc) / len(X)),
                     "mean": Xc.mean(axis=0), "var": Xc.var(axis=0)}
    return {"classes": classes, "params": params}


def score_gaussian(x, label, model):
    """log P(label) + sum_j log N(x_j ; mu_cj, sigma_cj^2)."""
    p = model["params"][label]
    x = np.asarray(x, dtype=float)
    log_pdf = -0.5 * np.log(2 * np.pi * p["var"]) - (x - p["mean"]) ** 2 / (2 * p["var"])
    return p["log_prior"] + float(log_pdf.sum())


def classify_gaussian(X, model):
    return [max(model["classes"], key=lambda c: score_gaussian(x, c, model)) for x in np.asarray(X)]


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    all_docs, all_labels = read_documents(os.path.join(here, "all_sentiment_shuffled.txt"))
    train_docs, eval_docs = split_80_20(all_docs)
    train_labels, eval_labels = split_80_20(all_labels)
    model = train_nb(train_docs, train_labels, alpha=1.0)
    for doc in (["great"], ["bad"], ["a", "top-quality", "performance"]):
        print(doc, {c: math.exp(score_doc_label(doc, c, model)) for c in model["classes"]}, classify_nb(doc, model))
    guesses = classify_documents(eval_docs, model)
    k = sum(g == t for g, t in zip(guesses, eval_labels))
    print("accuracy", accuracy(eval_labels, guesses), "95% CI", wilson_interval(k, len(eval_labels)))
