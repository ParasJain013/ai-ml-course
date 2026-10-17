# %% [markdown]
# # B2.4 Naive Bayes — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b2_4_naive_bayes_ex2.py   (CPU, ~5 s; needs the lab's download)

# %%
from __future__ import annotations

import re
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.special import logsumexp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
MAX_FP_RATE = 0.001
SMS_FILE = Path(__file__).resolve().parents[3] / "data" / "sms_spam" / "SMSSpamCollection"


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
toy = [("win cash now", 1), ("win a free prize now", 1), ("lunch now", 0), ("see you at lunch", 0),
       ("free for lunch", 0)]
V_TOY, N_TOT = 10, {0: 9, 1: 7}
counts = {0: {"cash": 0, "prize": 0, "at": 1, "lunch": 3, "win": 0},
          1: {"cash": 1, "prize": 1, "at": 0, "lunch": 0, "win": 2}}
prior = {0: Fraction(3, 5), 1: Fraction(2, 5)}
for alpha in [Fraction(1), Fraction(1, 10)]:
    joint = {}
    for c in (0, 1):
        th = [(counts[c][w] + alpha) / (N_TOT[c] + alpha * V_TOY) for w in ["cash", "prize", "at", "lunch"]]
        joint[c] = prior[c] * th[0] * th[1] * th[2] * th[3]
    p = joint[1] / (joint[0] + joint[1])
    print(f"alpha {float(alpha):>3}: ham {joint[0]} = {float(joint[0]):.4e}   spam {joint[1]} = {float(joint[1]):.4e}   "
          f"P(spam) {float(p):.4f}")
w_lunch = np.log((1 / 17) / (4 / 19))
w_win = np.log((3 / 17) / (1 / 19))
bias = np.log(0.4 / 0.6)
print(f"(c) weight lunch {w_lunch:.4f}   weight win {w_win:.4f}   bias log(0.4/0.6) {bias:.4f}")
for k in (1, 3):
    lo = bias + k * w_win
    print(f"(d) 'win' x {k}: log-odds {lo:.4f}   P(spam) {1 / (1 + np.exp(-lo)):.4f}")

# %%
section("data")
if not SMS_FILE.exists():
    raise FileNotFoundError("Run labs/B/b2_4_naive_bayes.py once first: it downloads the SMS collection.")
lines = SMS_FILE.read_text(encoding="utf-8").splitlines()
labels, texts_t = zip(*(line.split("\t", 1) for line in lines))
y_all = np.array([1 if lab == "spam" else 0 for lab in labels], dtype=np.int64)
texts = list(texts_t)
gen = np.random.default_rng(SEED)  # the lab's stratified 60/20/20 split
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y_all):
    rows = gen.permutation(np.flatnonzero(y_all == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
txt_tr, txt_va = [texts[i] for i in i_tr], [texts[i] for i in i_va]
y_tr, y_va = y_all[i_tr], y_all[i_va]
print(f"train {len(txt_tr)}  val {len(txt_va)}")

# %%
WORD_RE = re.compile(r"(?u)\b\w\w+\b")
NUM_RE = re.compile(r"\d+")


def tok_words(text: str) -> list[str]:
    """Lowercase, words of 2+ characters (CountVectorizer's rule)."""
    return WORD_RE.findall(text.lower())


def tok_num(text: str) -> list[str]:
    """As tok_words, after replacing every digit run with the placeholder 'numtoken'."""
    return WORD_RE.findall(NUM_RE.sub(" numtoken ", text.lower()))


class Vectorizer:
    """Vocabulary from the documents passed to fit; unknown words are dropped at transform time."""

    def __init__(self, tokenize: Callable[[str], list[str]]) -> None:
        self.tokenize = tokenize
        self.vocab: dict[str, int] = {}

    def fit(self, docs: list[str]) -> Vectorizer:
        self.vocab = {w: j for j, w in enumerate(sorted({w for d in docs for w in self.tokenize(d)}))}
        return self

    def transform(self, docs: list[str]) -> sparse.csr_matrix:
        rows, cols = [], []
        for i, d in enumerate(docs):
            for w in self.tokenize(d):
                if w in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[w])
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(self.vocab)))


def fit_nb(X: sparse.csr_matrix, y: IntArray, alpha: float) -> tuple[FloatArray, FloatArray]:
    """Log priors (2,) and smoothed log P(word | class) (2, V)."""
    log_prior = np.log(np.array([np.mean(y == 0), np.mean(y == 1)]))
    cnt = np.vstack([np.asarray(X[y == c].sum(axis=0)).ravel() for c in (0, 1)]) + alpha
    return log_prior, np.log(cnt) - np.log(cnt.sum(axis=1, keepdims=True))


def p_spam(X: sparse.csr_matrix, log_prior: FloatArray, log_theta: FloatArray) -> FloatArray:
    """P(spam | x) for each row. X (n, V) -> (n,)."""
    joint = X @ log_theta.T + log_prior  # (n, 2)
    return np.exp(joint[:, 1] - logsumexp(joint, axis=1))


def mean_log_loss(y: IntArray, p: FloatArray) -> float:
    """Binary log-loss, probabilities clipped as sklearn.metrics.log_loss does."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cv_log_loss(docs: list[str], y: IntArray, tokenize: Callable[[str], list[str]], alpha: float) -> float:
    """5-fold stratified CV log-loss; the vocabulary is rebuilt inside every fold."""
    losses = []
    for f_tr, f_va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(docs, y):
        vec = Vectorizer(tokenize).fit([docs[i] for i in f_tr])
        lp, lt = fit_nb(vec.transform([docs[i] for i in f_tr]), y[f_tr], alpha)
        losses.append(mean_log_loss(y[f_va], p_spam(vec.transform([docs[i] for i in f_va]), lp, lt)))
    return float(np.mean(losses))


def report(name: str, tokenize: Callable[[str], list[str]]) -> None:
    """Tune alpha by CV, then report validation metrics and the low-false-positive threshold."""
    cv_scores = {a: cv_log_loss(txt_tr, y_tr, tokenize, a) for a in ALPHAS}
    best = min(cv_scores, key=cv_scores.get)
    print(f"[{name}] CV log-loss by alpha: " + "  ".join(f"{a}: {s:.4f}" for a, s in cv_scores.items()))
    vec = Vectorizer(tokenize).fit(txt_tr)
    lp, lt = fit_nb(vec.transform(txt_tr), y_tr, best)
    p = p_spam(vec.transform(txt_va), lp, lt)
    pred = p >= 0.5
    tp, fp = np.sum(pred & (y_va == 1)), np.sum(pred & (y_va == 0))
    fn = np.sum(~pred & (y_va == 1))
    print(f"[{name}] alpha {best}: val accuracy {np.mean(pred == y_va):.4f}  precision {tp / (tp + fp):.4f}  "
          f"recall {tp / (tp + fn):.4f}  log-loss {mean_log_loss(y_va, p):.4f}")
    ham_scores = np.sort(p[y_va == 0])[::-1]  # highest-scoring ham first
    allowed_fp = int(np.floor(MAX_FP_RATE * len(ham_scores)))  # 0 for 965 ham messages
    thr = np.nextafter(ham_scores[allowed_fp], 1.0)  # just above the (allowed_fp + 1)-th highest ham score
    flagged = p >= thr
    print(f"[{name}] threshold {thr:.6f} (FP allowed {allowed_fp} of {len(ham_scores)} ham): "
          f"FP {np.sum(flagged & (y_va == 0))}  spam recall {np.mean(flagged[y_va == 1]):.4f}")
    if name == "words":  # check our model against scikit-learn once
        cvz = CountVectorizer()
        nb = MultinomialNB(alpha=best).fit(cvz.fit_transform(txt_tr), y_tr)
        np.testing.assert_allclose(nb.predict_proba(cvz.transform(txt_va))[:, 1], p, rtol=1e-8, atol=1e-12)
        print(f"[{name}] matches CountVectorizer + MultinomialNB: OK")


section("from-scratch spam filter")
report("words", tok_words)
report("numbers -> one token", tok_num)

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
