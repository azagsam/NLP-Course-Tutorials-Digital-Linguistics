"""
Text Representation Techniques Assignment (Solutions Version)
===========================================================

Implement core text-representation methods for spam classification.

Covers:
- text preprocessing (tokenisation, normalisation, stop-word removal)
- vocabulary construction
- Bag-of-Words (Count Vectorizer) from scratch
- TF-IDF from scratch (with data-leakage-safe IDF)
- hand-crafted feature engineering

"""

import math
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("assignments/data/spam.csv")

STOP_WORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need dare ought used "
    "i me my we our you your he she it his her its they them their "
    "this that these those and but or nor so yet for of in on at to "
    "by with from up out about into through during before after above "
    "below between each few more most other some such no than then "
    "too very just here there what which who whom when where why how "
    "all both each few more most other some such only own same so "
    "than too s t ll re ve d m".split()
)


# ---------------------------------------------------------------------------
# Step 1 – Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Load the SMS spam CSV and return (texts, labels) where label 1 = spam."""
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    texts = df["text"].astype(str).tolist()
    labels = (df["label"].str.lower() == "spam").astype(int).tolist()
    return texts, labels


# ---------------------------------------------------------------------------
# Step 2 – Preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Normalise and tokenise a single SMS message.

    1. Lowercase.
    2. Remove punctuation and digits (keep alphabetic only).
    3. Split on whitespace.
    4. Optionally filter stop words.
    5. Filter single-character tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


# ---------------------------------------------------------------------------
# Step 3 – Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(corpus: list[str], max_vocab: int = 3000) -> dict[str, int]:
    """
    Build a token -> index mapping from the training corpus.

    1. Preprocess every document.
    2. Count token frequencies.
    3. Keep the top max_vocab tokens.
    4. Return {token: index} ordered by descending frequency.
    """
    counter: Counter = Counter()
    for doc in corpus:
        counter.update(preprocess(doc))
    vocab = {token: idx for idx, (token, _) in enumerate(counter.most_common(max_vocab))}
    return vocab


# ---------------------------------------------------------------------------
# Step 4 – Bag-of-Words (Count Vectorizer)
# ---------------------------------------------------------------------------

def count_vectorize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    """
    Convert texts to a count (Bag-of-Words) matrix.

    Returns numpy array of shape (n_docs, vocab_size), dtype int32.
    """
    n_docs = len(texts)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.int32)

    for i, doc in enumerate(texts):
        for token in preprocess(doc):
            if token in vocab:
                matrix[i, vocab[token]] += 1

    return matrix


# ---------------------------------------------------------------------------
# Step 5 – TF-IDF
# ---------------------------------------------------------------------------

def compute_idf(corpus_tokens: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    """
    Compute the IDF vector for all vocabulary tokens.

    Formula (sklearn-style, smoothed):
        IDF(t) = log((1 + N) / (1 + df(t))) + 1
    """
    n_docs = len(corpus_tokens)
    vocab_size = len(vocab)
    df = np.zeros(vocab_size, dtype=np.float64)

    for tokens in corpus_tokens:
        for token in set(tokens):        # count each token once per doc
            if token in vocab:
                df[vocab[token]] += 1

    idf = np.log((1 + n_docs) / (1 + df)) + 1.0
    return idf


def tfidf_vectorize(
    texts: list[str],
    vocab: dict[str, int],
    idf: np.ndarray,
) -> np.ndarray:
    """
    Convert texts to a TF-IDF matrix.

        TF(t, d)     = count(t, d) / len(d)
        TF-IDF(t, d) = TF(t, d) * IDF(t)

    IDF is passed in (pre-computed on training data) to prevent leakage.

    Returns numpy array of shape (n_docs, vocab_size), dtype float64.
    """
    n_docs = len(texts)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

    for i, doc in enumerate(texts):
        tokens = preprocess(doc)
        if not tokens:
            continue
        token_counts = Counter(tokens)
        n_tokens = len(tokens)
        for token, count in token_counts.items():
            if token in vocab:
                tf = count / n_tokens
                matrix[i, vocab[token]] = tf * idf[vocab[token]]

    return matrix


# ---------------------------------------------------------------------------
# Step 6 – Custom hand-crafted features
# ---------------------------------------------------------------------------

_SPAM_WORDS = frozenset(
    "free win winner won cash prize claim urgent call txt text reply "
    "guaranteed offer discount limited mobile ringtone download bonus "
    "selected congratulations awarded voucher reward collect".split()
)


def extract_custom_features(text: str) -> list[float]:
    """
    Hand-crafted feature vector for one SMS message (9 features).
    Suggested features include the list below, but feel free to be creative and experiment with your own ideas!

    1.  Message length (characters)
    2.  Number of tokens
    3.  Number of digits
    4.  Number of uppercase characters
    5.  Uppercase character ratio
    6.  Number of punctuation marks (!, ?, .)
    7.  Number of currency symbols ($, £, €)
    8.  Count of spam-indicator words (lowercased)
    9.  Type-token ratio  (unique tokens / total tokens)
    """
    tokens = text.split()
    n_chars = len(text) or 1
    n_tokens = len(tokens) or 1

    n_digits = sum(c.isdigit() for c in text)
    n_upper = sum(c.isupper() for c in text)
    upper_ratio = n_upper / n_chars
    n_punct = text.count("!") + text.count("?") + text.count(".")
    n_currency = text.count("$") + text.count("£") + text.count("€")

    lower_tokens = set(t.lower().strip(string.punctuation) for t in tokens)
    n_spam_words = len(lower_tokens & _SPAM_WORDS)

    ttr = len(set(t.lower() for t in tokens)) / n_tokens

    return [
        float(n_chars),
        float(n_tokens),
        float(n_digits),
        float(n_upper),
        upper_ratio,
        float(n_punct),
        float(n_currency),
        float(n_spam_words),
        ttr,
    ]


# ---------------------------------------------------------------------------
# Step 7 – Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[int],
    y_test: list[int],
    label: str,
) -> None:
    """Train logistic regression and print a classification report."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=4))
    print(f"  Macro F1: {f1_macro:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading dataset …")
    texts, labels = load_dataset(DATA_PATH)
    print(f"  Total samples : {len(texts)}")
    print(f"  Spam          : {sum(labels)}  ({sum(labels)/len(labels):.1%})")
    print(f"  Ham           : {len(labels)-sum(labels)}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.5, random_state=42, stratify=labels
    )

    # Vocabulary (built on training data only to avoid leakage)
    vocab = build_vocabulary(X_train_raw, max_vocab=3000)
    print(f"\nVocabulary size : {len(vocab)}")

    # --- Bag-of-Words ---
    X_train_bow = count_vectorize(X_train_raw, vocab)
    X_test_bow = count_vectorize(X_test_raw, vocab)
    evaluate(X_train_bow, X_test_bow, y_train, y_test, "Bag-of-Words (Count)")

    # --- TF-IDF ---
    train_tokens = [preprocess(t) for t in X_train_raw]
    idf = compute_idf(train_tokens, vocab)
    X_train_tfidf = tfidf_vectorize(X_train_raw, vocab, idf)
    X_test_tfidf = tfidf_vectorize(X_test_raw, vocab, idf)
    evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF (from scratch)")

    # --- Custom features ---
    custom_train = np.array([extract_custom_features(t) for t in X_train_raw])
    custom_test = np.array([extract_custom_features(t) for t in X_test_raw])
    evaluate(custom_train, custom_test, y_train, y_test, "Custom Hand-crafted Features")


if __name__ == "__main__":
    main()
