"""
Text Representation Techniques Assignment (Student Version)
===========================================================

Implement core text-representation methods for spam classification.

Covers:
- text preprocessing (tokenisation, normalisation, stop-word removal)
- vocabulary construction
- Bag-of-Words (Count Vectorizer) from scratch
- TF-IDF from scratch (with data-leakage-safe IDF)
- hand-crafted feature engineering

Complete all TODO sections.
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
# Step 1 - Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Load the SMS spam CSV and return (texts, labels) where label 1 = spam."""
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    texts = df["text"].astype(str).tolist()
    labels = (df["label"].str.lower() == "spam").astype(int).tolist()
    return texts, labels


# ---------------------------------------------------------------------------
# Step 2 - Preprocessing
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
    # TODO: lowercase text

    # TODO: remove punctuation and digits
    
    # TODO: split into tokens
    tokens: list[str] = []

    # TODO: if remove_stopwords is True, remove tokens from STOP_WORDS
    
    # TODO: remove single-character tokens
    return tokens


# ---------------------------------------------------------------------------
# Step 3 - Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(corpus: list[str], max_vocab: int = 3000) -> dict[str, int]:
    """
    Build a token -> index mapping from the training corpus.

    1. Preprocess every document.
    2. Count token frequencies.
    3. Keep the top max_vocab tokens.
    4. Return {token: index} ordered by descending frequency.
    """
    # TODO: create a vocabulary
    
    return {}


# ---------------------------------------------------------------------------
# Step 4 - Bag-of-Words (Count Vectorizer)
# ---------------------------------------------------------------------------

def count_vectorize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    """
    Convert texts to a count (Bag-of-Words) matrix.

    Returns numpy array of shape (n_docs, vocab_size), dtype int32.
    """
    n_docs = len(texts)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.int32)

    # TODO: for each document:
    # - preprocess document
    # - increment matrix[i, vocab[token]] for each token found in vocab

    return matrix


# ---------------------------------------------------------------------------
# Step 5 - TF-IDF
# ---------------------------------------------------------------------------

def compute_idf(corpus_tokens: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    """
    Compute the IDF vector for all vocabulary tokens.

    Formula (smoothed):
        IDF(t) = log((1 + N) / (1 + df(t))) + 1
    """
    n_docs = len(corpus_tokens)
    vocab_size = len(vocab)
    df = np.zeros(vocab_size, dtype=np.float64)

    # TODO: compute document frequency per token
    # Hint: use set(tokens) so each token counts once per document

    # TODO: compute and return smoothed IDF vector
    
    pass


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

    pass


# ---------------------------------------------------------------------------
# Step 6 - Custom hand-crafted features
# ---------------------------------------------------------------------------

_SPAM_WORDS = frozenset(
    "free win winner won cash prize claim urgent call txt text reply "
    "guaranteed offer discount limited mobile ringtone download bonus "
    "selected congratulations awarded voucher reward collect".split()
)


def extract_custom_features(text: str) -> list[float]:
    """
    Hand-crafted feature vector for one SMS message (9 features).

    Suggested features:
    1. Message length (characters)
    2. Number of tokens
    3. Number of digits
    4. Number of uppercase characters
    5. Uppercase character ratio
    6. Number of punctuation marks (!, ?, .)
    7. Number of currency symbols ($, GBP, EUR)
    8. Count of spam-indicator words (lowercased)
    9. Type-token ratio (unique tokens / total tokens)
    """
    # TODO: compute all 9 features and return as list[float]
    # Tip: protect against division by zero for empty text/tokens.
    
    pass


# ---------------------------------------------------------------------------
# Step 7 - Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[int],
    y_test: list[int],
    label: str,
) -> None:
    """Train logistic regression and print a classification report."""
    # TODO: train LogisticRegression(max_iter=1000, random_state=42)
    # TODO: predict on X_test
    # TODO: compute macro F1
    # TODO: print section header and classification_report
    


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading dataset...")
    texts, labels = load_dataset(DATA_PATH)

    print(f"  Total samples : {len(texts)}")
    print(f"  Spam          : {sum(labels)}  ({sum(labels)/len(labels):.1%})")
    print(f"  Ham           : {len(labels)-sum(labels)}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.5, random_state=42, stratify=labels
    )

    # TODO: build vocabulary from X_train_raw only (avoid leakage)
    vocab: dict[str, int] = {}

    # --- Bag-of-Words ---
    # TODO: vectorize train/test with count_vectorize and evaluate

    # --- TF-IDF ---
    # TODO: preprocess X_train_raw -> tokens
    # TODO: compute idf on training tokens only
    # TODO: vectorize train/test with tfidf_vectorize and evaluate

    # --- Custom features ---
    # TODO: build numpy feature matrices from extract_custom_features
    # TODO: evaluate custom features

    # --- Custom approach (optional) ---
    # TODO: add your own representation idea and evaluate it 
    # (e.g., use scikit-learn tf-idf implementation and try to find the best hyperparameter values)
    # (run script using machine learning models)


if __name__ == "__main__":
    main()

