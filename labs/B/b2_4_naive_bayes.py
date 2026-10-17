# %% [markdown]
# # B2.4 Naive Bayes — lab
# Lesson: lessons/B/b2_4_naive_bayes.md
# Run: uv run python labs/B/b2_4_naive_bayes.py   (CPU, defaults: ~10 s; first run downloads 0.5 MB)
# GPU: no

# %%
from __future__ import annotations

import hashlib
import io
import re
import sys
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.special import logsumexp
from sklearn.datasets import load_wine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.naive_bayes import GaussianNB, MultinomialNB

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "b2_4"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU (or more data): the whole SMS collection is used either way.
GPU_SCALE = False
ALPHA = 1.0  # Laplace (add-one) smoothing
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# The SMS Spam Collection (Almeida, Gomez Hidalgo & Yamakami, 2011), UCI ML Repository, CC BY 4.0
# (VERIFY licence and links, as of 2026-10). The mirror is a byte-identical copy of the UCI file;
# the checksum guarantees everyone runs on the same 5,574 messages.
SMS_DIR = DATA / "sms_spam"
SMS_FILE = SMS_DIR / "SMSSpamCollection"
SMS_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
SMS_MIRROR = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
SMS_UCI_ZIP = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# %%
# Data: download once into data/sms_spam/ (idempotent), check the checksum, parse "label<TAB>text".
section("data")


def download_sms() -> Path:
    """Fetch the SMS Spam Collection into data/sms_spam/ unless it is already there; verify it."""
    if not SMS_FILE.exists():
        SMS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            raw = urllib.request.urlopen(SMS_MIRROR, timeout=60).read()
            source = SMS_MIRROR
        except urllib.error.URLError as err:  # mirror down: fall back to the original zip on UCI
            print(f"~ mirror failed ({err}); trying UCI")
            with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(SMS_UCI_ZIP, timeout=60).read())) as zf:
                raw = zf.read("SMSSpamCollection")
            source = SMS_UCI_ZIP
        SMS_FILE.write_bytes(raw)
        print(f"~ downloaded {len(raw):,} bytes from {source}")  # first run only, hence ~
    digest = hashlib.sha256(SMS_FILE.read_bytes()).hexdigest()
    if digest != SMS_SHA256:
        raise ValueError(f"{SMS_FILE} has sha256 {digest}, expected {SMS_SHA256}; delete it and rerun")
    return SMS_FILE


lines = download_sms().read_text(encoding="utf-8").splitlines()
labels, texts = zip(*(line.split("\t", 1) for line in lines))
y_all = np.array([1 if lab == "spam" else 0 for lab in labels], dtype=np.int64)  # 1 = spam
texts_all = list(texts)
print(f"{SMS_FILE.relative_to(ROOT)}: checksum OK, {len(texts_all)} messages, spam share {y_all.mean():.4f}")
print(f"example ham:  {texts_all[1]!r}")
print(f"example spam: {texts_all[2][:70]!r}...")


def train_val_test_split(
    n: int, val_frac: float, test_frac: float, gen: np.random.Generator, stratify: IntArray | None = None
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py
    groups = [np.arange(n)] if stratify is None else [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    parts: list[list[IntArray]] = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)
        n_test, n_val = round(test_frac * len(rows)), round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    train, val, test = (gen.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), y_all)
txt_tr, txt_va = [texts_all[i] for i in i_tr], [texts_all[i] for i in i_va]
y_tr, y_va = y_all[i_tr], y_all[i_va]
print(f"train {len(txt_tr)}  val {len(txt_va)}  (test {len(i_te)} kept aside)   "
      f"spam: train {y_tr.sum()}  val {y_va.sum()}")
print(f"majority-class baseline (always ham) on val: accuracy {1 - y_va.mean():.4f}")

# %%
# Worked example by hand: five tiny messages, Laplace smoothing, one query.
section("worked example by hand")
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # CountVectorizer's default: words of 2+ letters/digits


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words of two or more characters (as CountVectorizer does)."""
    return TOKEN_RE.findall(text.lower())


toy = [("win cash now", 1), ("win a free prize now", 1), ("lunch now", 0), ("see you at lunch", 0),
       ("free for lunch", 0)]
toy_vocab = sorted({w for t, _ in toy for w in tokenize(t)})
print(f"vocabulary ({len(toy_vocab)}): {toy_vocab}   ('a' is dropped: one letter)")
query = tokenize("free lunch now win")
toy_scores = {}
for c, name in [(0, "ham"), (1, "spam")]:
    counts = Counter(w for t, lab in toy if lab == c for w in tokenize(t))
    total = sum(counts.values())
    prior = sum(lab == c for _, lab in toy) / len(toy)
    theta = {w: (counts[w] + ALPHA) / (total + ALPHA * len(toy_vocab)) for w in query}
    toy_scores[name] = np.log(prior) + sum(np.log(theta[w]) for w in query)
    print(f"{name}: prior {prior}  word count {total}  P(w | {name}) = (count + 1) / ({total} + {len(toy_vocab)}): "
          + "  ".join(f"{w} {counts[w]}->{theta[w]:.4f}" for w in query))
    print(f"      log score = log {prior} + sum log P(w | {name}) = {toy_scores[name]:.4f}   "
          f"(product {np.exp(toy_scores[name]):.4e})")
p_spam_toy = 1.0 / (1.0 + np.exp(toy_scores["ham"] - toy_scores["spam"]))
print(f"P(spam | 'free lunch now win') = {p_spam_toy:.4f}")
print("without smoothing: P(win | ham) = 0/9 and P(lunch | spam) = 0/7, so both classes score 0 (log = -inf)")

# %%
# From scratch: vocabulary from the training messages only, count matrix, smoothed log-probabilities.
section("from scratch: multinomial naive Bayes")


def build_vocab(docs: list[str]) -> dict[str, int]:
    """Map each word seen in the training documents to a column index (alphabetical)."""
    return {w: j for j, w in enumerate(sorted({w for doc in docs for w in tokenize(doc)}))}


def count_matrix(docs: list[str], vocab: dict[str, int]) -> sparse.csr_matrix:
    """Bag of words: row i, column j = times word j occurs in doc i; unknown words are dropped. -> (n, V)."""
    rows, cols = [], []
    for i, doc in enumerate(docs):
        for w in tokenize(doc):
            if w in vocab:
                rows.append(i)
                cols.append(vocab[w])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(vocab)))


def fit_multinomial_nb(Xc: sparse.csr_matrix, y: IntArray, alpha: float) -> tuple[FloatArray, FloatArray]:
    """Log priors (K,) and smoothed log P(word | class) (K, V) from a count matrix Xc (n, V)."""
    classes = np.unique(y)
    log_prior = np.log(np.array([np.mean(y == c) for c in classes]))  # (K,)
    word_counts = np.vstack([np.asarray(Xc[y == c].sum(axis=0)).ravel() for c in classes])  # (K, V)
    smoothed = word_counts + alpha
    log_theta = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))  # (K, V), rows sum to 1 in prob
    return log_prior, log_theta


def predict_log_proba_nb(Xc: sparse.csr_matrix, log_prior: FloatArray, log_theta: FloatArray) -> FloatArray:
    """Log posteriors via Bayes' rule, normalized with log-sum-exp. Xc (n, V) -> (n, K)."""
    joint = Xc @ log_theta.T + log_prior  # (n, V) @ (V, K) -> (n, K): log P(c) + sum_w x_w log P(w | c)
    return joint - logsumexp(joint, axis=1, keepdims=True)  # subtract log P(x): rows sum to 1 in prob


vocab = build_vocab(txt_tr)
Xc_tr, Xc_va = count_matrix(txt_tr, vocab), count_matrix(txt_va, vocab)
log_prior, log_theta = fit_multinomial_nb(Xc_tr, y_tr, ALPHA)
logp_va = predict_log_proba_nb(Xc_va, log_prior, log_theta)
pred_va = logp_va.argmax(axis=1)
tn, fp, fn, tp = confusion_matrix(y_va, pred_va).ravel()
print(f"vocabulary {len(vocab)} words   train count matrix {Xc_tr.shape} with {Xc_tr.nnz} non-zeros "
      f"({100 * Xc_tr.nnz / np.prod(Xc_tr.shape):.2f}% filled)")
print(f"priors: P(ham) {np.exp(log_prior[0]):.4f}  P(spam) {np.exp(log_prior[1]):.4f}")
unseen = sum(w not in vocab for doc in txt_va for w in tokenize(doc))
print(f"val words not in the training vocabulary (ignored): {unseen} of {sum(len(tokenize(d)) for d in txt_va)}")
print(f"val accuracy {np.mean(pred_va == y_va):.4f}   confusion: TN {tn}  FP {fp}  FN {fn}  TP {tp}")
print(f"spam precision {tp / (tp + fp):.4f}  spam recall {tp / (tp + fn):.4f}")
ratio = log_theta[1] - log_theta[0]  # (V,) log P(w | spam) / P(w | ham)
words = np.array(sorted(vocab, key=vocab.get))
print("most spam-indicating words:", words[np.argsort(ratio)[::-1][:10]].tolist())
print("most ham-indicating words: ", words[np.argsort(ratio)[:10]].tolist())

# %%
# With scikit-learn: CountVectorizer + MultinomialNB give the same vocabulary, counts and probabilities.
section("with scikit-learn")
cv = CountVectorizer()  # lowercase=True and token_pattern r"(?u)\b\w\w+\b" by default (VERIFY as of 2026-10)
Xs_tr, Xs_va = cv.fit_transform(txt_tr), cv.transform(txt_va)
assert list(cv.get_feature_names_out()) == list(words)
assert (Xs_tr != Xc_tr).nnz == 0
print("CountVectorizer vocabulary and counts identical to ours: OK")
nb = MultinomialNB(alpha=ALPHA).fit(Xs_tr, y_tr)
np.testing.assert_allclose(nb.feature_log_prob_, log_theta, rtol=1e-10)
np.testing.assert_allclose(nb.class_log_prior_, log_prior, rtol=1e-10)
np.testing.assert_allclose(nb.predict_log_proba(Xs_va), logp_va, rtol=1e-8, atol=1e-10)
print("MultinomialNB(alpha=1) log-probabilities identical to ours: OK")
p_nb = np.exp(logp_va[:, 1])  # P(spam)
lr = LogisticRegression(C=10.0, max_iter=5_000).fit(Xs_tr, y_tr)  # same counts, discriminative model
p_lr = lr.predict_proba(Xs_va)[:, 1]
print(f"naive Bayes:          val accuracy {np.mean((p_nb >= 0.5) == y_va):.4f}   val log-loss {log_loss(y_va, p_nb):.4f}")
print(f"logistic regression:  val accuracy {np.mean((p_lr >= 0.5) == y_va):.4f}   val log-loss {log_loss(y_va, p_lr):.4f}")
extreme = (p_nb < 1e-3) | (p_nb > 1 - 1e-3)
print(f"naive Bayes puts {extreme.mean():.4f} of val messages below 0.001 or above 0.999; "
      f"accuracy on those {np.mean((p_nb[extreme] >= 0.5) == y_va[extreme]):.4f}")
wrong = np.flatnonzero((p_nb >= 0.5) != y_va)
worst = wrong[np.argmax(np.abs(p_nb[wrong] - 0.5))]
print(f"most confident mistake: true {'spam' if y_va[worst] else 'ham'}, P(spam) {p_nb[worst]:.1e}: {txt_va[worst][:80]!r}")
fit_s = time.perf_counter()
for _ in range(20):
    MultinomialNB(alpha=ALPHA).fit(Xs_tr, y_tr)
print(f"~ MultinomialNB fit time {(time.perf_counter() - fit_s) / 20 * 1000:.2f} ms")

# %%
# Gaussian naive Bayes from scratch on the wine data: one normal per (class, feature).
section("from scratch: Gaussian naive Bayes on wine")
wine = load_wine()
Xw, yw = wine.data.astype(np.float64), wine.target.astype(np.int64)
iw_tr, iw_va, _ = train_val_test_split(len(yw), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), yw)


def fit_gaussian_nb(X: FloatArray, y: IntArray, var_smoothing: float = 1e-9) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-class log priors (K,), means (K, d) and variances (K, d); variances get a small floor."""
    classes = np.unique(y)
    log_prior = np.log(np.array([np.mean(y == c) for c in classes]))
    means = np.vstack([X[y == c].mean(axis=0) for c in classes])  # (K, d)
    var = np.vstack([X[y == c].var(axis=0) for c in classes])  # (K, d) MLE: divides by n_c
    var += var_smoothing * X.var(axis=0).max()  # scikit-learn's floor: no zero variances
    return log_prior, means, var


def predict_log_proba_gnb(X: FloatArray, log_prior: FloatArray, means: FloatArray, var: FloatArray) -> FloatArray:
    """Sum of per-feature normal log densities plus log prior, normalized. X (n, d) -> (n, K)."""
    diff = X[:, None, :] - means[None, :, :]  # (n, K, d)
    log_lik = -0.5 * (np.log(2 * np.pi * var)[None] + diff**2 / var[None]).sum(axis=2)  # (n, K)
    joint = log_lik + log_prior
    return joint - logsumexp(joint, axis=1, keepdims=True)


gp, gm, gv = fit_gaussian_nb(Xw[iw_tr], yw[iw_tr])
lp_w = predict_log_proba_gnb(Xw[iw_va], gp, gm, gv)
gnb = GaussianNB().fit(Xw[iw_tr], yw[iw_tr])
np.testing.assert_allclose(gnb.theta_, gm, rtol=1e-12)
np.testing.assert_allclose(gnb.var_, gv, rtol=1e-12)
np.testing.assert_allclose(gnb.predict_log_proba(Xw[iw_va]), lp_w, rtol=1e-8, atol=1e-8)
print(f"class means of proline: {np.round(gm[:, wine.feature_names.index('proline')], 1).tolist()}")
print(f"val accuracy {np.mean(lp_w.argmax(axis=1) == yw[iw_va]):.4f}   matches GaussianNB (means, variances, "
      f"log-probabilities): OK")
corr = np.corrcoef(Xw[iw_tr][yw[iw_tr] == 0].T)
upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
print(f"within class 0, |correlation| between features: median {np.median(upper):.4f}  max {upper.max():.4f} "
      f"(naive Bayes assumes 0)")

# %%
# Key plot: naive Bayes is accurate but overconfident. Posterior log-odds of the validation messages.
section("key plot")
log_odds = logp_va[:, 1] - logp_va[:, 0]  # (n_val,) log P(spam | x) / P(ham | x)
print(f"validation log-odds: median for ham {np.median(log_odds[y_va == 0]):.2f}  "
      f"median for spam {np.median(log_odds[y_va == 1]):.2f}")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
bins = np.linspace(-60, 60, 61)
ax.hist(np.clip(log_odds[y_va == 0], -60, 60), bins=bins, histtype="step", lw=1.5, label="ham (true)")
ax.hist(np.clip(log_odds[y_va == 1], -60, 60), bins=bins, histtype="stepfilled", alpha=0.5, hatch="//",
        color="tab:orange", label="spam (true)")
ax.set_yscale("log")
ax.axvspan(-6.9, 6.9, color="0.9", zorder=0, label="0.001 < p < 0.999")
ax.set_xlabel("naive Bayes log-odds of spam (clipped to ±60)")
ax.set_ylabel("validation messages (log scale)")
ax.set_title("Naive Bayes separates well but is wildly overconfident")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "b2_4_nb_log_odds.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b2_4_nb_log_odds.png")

# %%
# Try this (not graded):
# 1. Set ALPHA to 0.01 and 10. How do accuracy, spam recall and log-loss change?
# 2. Use CountVectorizer(binary=True) (does a word occur at all, not how often) and compare.
# 3. Duplicate every word of each message (doc + " " + doc). What happens to the log-odds, and why?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
