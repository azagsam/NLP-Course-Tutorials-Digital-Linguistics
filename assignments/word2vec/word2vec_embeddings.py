"""
Word Embeddings Assignment (Student Version)
============================================

Explore pre-trained word embeddings (GloVe 100-d) through five tasks that
build geometric intuition, reveal linguistic structure, and surface ethical
concerns embedded in real-world corpora.

Tasks
-----
1. Word Similarity & Nearest Neighbours
   Implement cosine similarity from scratch and find semantic neighbours.

2. Word Analogy Task
   Solve "a : b :: c : ?" via vector arithmetic. Evaluate on a benchmark.

3. Visualising Embedding Space
   Project high-dimensional vectors to 2-D with PCA and t-SNE and plot
   labelled semantic clusters.

4. Sentence Representation via Averaging
   Aggregate word vectors into document embeddings; use them for
   classification and discuss the method's limits.

5. Bias Detection
   Measure gender associations latent in the embeddings using a structured
   similarity test.

Complete every TODO section.
Run from the repo root:

    python assignments/word2vec/word2vec_embeddings.py

Dependencies: gensim, numpy, scikit-learn, pandas, matplotlib
Model download (~134 MB, one-time): glove-wiki-gigaword-100
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("assignments/data/spam.csv")
VIZ_PATH  = Path("assignments/word2vec/embedding_space.png")

# ---------------------------------------------------------------------------
# Helpers  (already implemented — do not modify)
# ---------------------------------------------------------------------------

def load_spam_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Load SMS spam CSV → (texts, labels), label 1 = spam."""
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    texts  = df["text"].astype(str).tolist()
    labels = (df["label"].str.lower() == "spam").astype(int).tolist()
    return texts, labels


def simple_tokenize(text: str) -> list[str]:
    """Lowercase, keep only alphabetic tokens."""
    return [t for t in text.lower().split() if t.isalpha()]


# ===========================================================================
# Step 1 – Load pre-trained embeddings
# ===========================================================================

def load_embeddings(model_name: str = "glove-wiki-gigaword-100") -> KeyedVectors:
    """
    Download (first run only) and return the GloVe KeyedVectors model.

    Useful API
    ----------
    kv = api.load(model_name)   → KeyedVectors instance
    kv[word]                    → np.ndarray of shape (100,)
    word in kv                  → bool
    kv.index_to_key             → list of all vocabulary words

    Returns
    -------
    KeyedVectors
    """
    # TODO: call api.load(model_name) and return the result
    pass


# ===========================================================================
# TASK 1 — Word Similarity & Nearest Neighbours
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 2 – Cosine similarity from scratch
# ---------------------------------------------------------------------------

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

        cos(u, v) = (u · v) / (||u|| · ||v||)

    Return 0.0 if either vector has zero norm (avoids division by zero).

    Parameters
    ----------
    u, v : np.ndarray  shape (d,)

    Returns
    -------
    float in [-1, 1]
    """
    # TODO: compute and return cosine similarity
    pass


# ---------------------------------------------------------------------------
# Step 3 – k-Nearest-Neighbour search (no built-ins allowed)
# ---------------------------------------------------------------------------

def most_similar_words(
    word: str,
    kv: KeyedVectors,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Find the top_n words closest to `word` in cosine similarity.

    Rules
    -----
    - Do NOT call kv.most_similar or any other Gensim similarity helper.
    - Use your cosine_similarity implementation above.
    - Exclude `word` itself from the result list.
    - Return pairs sorted by descending similarity.

    Parameters
    ----------
    word  : query word (must exist in kv)
    kv    : KeyedVectors
    top_n : how many neighbours to return

    Returns
    -------
    list[tuple[str, float]]  length == top_n
    """
    if word not in kv:
        raise KeyError(f"'{word}' not in vocabulary")

    # TODO: iterate kv.index_to_key, compute similarities, filter, sort
    pass


# ---------------------------------------------------------------------------
# Extension 1a – Polysemous words
# ---------------------------------------------------------------------------
# Some words have multiple meanings.  Inspect what the embedding captures for
# ambiguous words like "bank" (financial vs river) or "apple" (fruit vs brand).
#
# Question for reflection:
#   Look at the top-10 neighbours of "bank".  Which meaning dominates?
#   Does the ordering change if you query "river bank" by averaging the two
#   word vectors?  Implement it below and compare.

def average_query(words: list[str], kv: KeyedVectors) -> np.ndarray:
    """
    Return the element-wise mean of the vectors for `words`.
    Skip any word not in kv.
    Return a zero vector if none of the words are in kv.

    Parameters
    ----------
    words : list of query words
    kv    : KeyedVectors

    Returns
    -------
    np.ndarray  shape (kv.vector_size,)
    """
    # TODO: collect vectors for in-vocabulary words and return their mean
    pass


def most_similar_to_vector(
    query_vec: np.ndarray,
    kv: KeyedVectors,
    exclude: set[str] | None = None,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Find the top_n vocabulary words most similar to an arbitrary vector.

    Same rules as most_similar_words — use your cosine_similarity.
    Words in `exclude` are skipped.

    Parameters
    ----------
    query_vec : np.ndarray  shape (kv.vector_size,)
    kv        : KeyedVectors
    exclude   : set of words to ignore (e.g. the original query words)
    top_n     : int

    Returns
    -------
    list[tuple[str, float]]  length == top_n
    """
    exclude = exclude or set()
    # TODO: iterate vocabulary, compute similarities, skip excluded words, sort
    pass


# ---------------------------------------------------------------------------
# Extension 1b – Compare two embedding models  (optional)
# ---------------------------------------------------------------------------
# Load a second model (e.g. "glove-twitter-25") and compare nearest neighbours
# for the same query word.  What differs between Wikipedia and Twitter corpora?
#
# You can implement this directly in main() — no skeleton needed.


# ===========================================================================
# TASK 2 — Word Analogy
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 4 – Single analogy from vector arithmetic
# ---------------------------------------------------------------------------

def solve_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    kv: KeyedVectors,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    Solve:  word_a : word_b  ::  word_c : ???

    Classic 3CosAdd formula:
        query = kv[word_b] - kv[word_a] + kv[word_c]

    Find top_n words closest to `query`, excluding word_a, word_b, word_c.
    Do NOT use built-in analogy helpers.

    Parameters
    ----------
    word_a, word_b, word_c : str  (all must be in kv)
    kv                     : KeyedVectors
    top_n                  : int

    Returns
    -------
    list[tuple[str, float]]  length == top_n
    """
    for w in (word_a, word_b, word_c):
        if w not in kv:
            raise KeyError(f"'{w}' not in vocabulary")

    # TODO: compute query vector  →  kv[word_b] - kv[word_a] + kv[word_c]
    query_vector: np.ndarray | None = None

    # TODO: find top_n nearest words to query_vector (excluding the three inputs)
    pass


# ---------------------------------------------------------------------------
# Step 4b – Batch analogy evaluation
# ---------------------------------------------------------------------------

ANALOGY_BENCHMARK: list[tuple[str, str, str, str]] = [
    # (word_a, word_b, word_c, expected_answer)
    # Semantic – capitals
    ("paris",    "france",   "rome",    "italy"),
    ("paris",    "france",   "berlin",  "germany"),
    ("paris",    "france",   "madrid",  "spain"),
    # Semantic – royalty / gender
    ("man",      "king",     "woman",   "queen"),
    ("man",      "actor",    "woman",   "actress"),
    ("man",      "father",   "woman",   "mother"),
    # Syntactic – comparative / superlative
    ("big",      "bigger",   "small",   "smaller"),
    ("good",     "best",     "bad",     "worst"),
    ("fast",     "faster",   "slow",    "slower"),
    # Syntactic – verb tense
    ("walk",     "walked",   "run",     "ran"),
    ("swim",     "swam",     "fly",     "flew"),
    ("go",       "went",     "come",    "came"),
]


def evaluate_analogies(
    benchmark: list[tuple[str, str, str, str]],
    kv: KeyedVectors,
) -> tuple[float, list[dict]]:
    """
    Evaluate solve_analogy on every entry in `benchmark`.

    A prediction is CORRECT if the expected answer is the top-1 result.

    Parameters
    ----------
    benchmark : list of (word_a, word_b, word_c, expected)
    kv        : KeyedVectors

    Returns
    -------
    accuracy : float   (correct / total)
    details  : list of dicts with keys
               'analogy', 'expected', 'predicted', 'correct'
    """
    # TODO: for each (a, b, c, expected) in benchmark:
    #   - call solve_analogy(a, b, c, kv, top_n=1)
    #   - check whether expected == result[0][0]
    #   - collect into details list
    # TODO: compute accuracy = correct_count / len(benchmark)
    # TODO: return (accuracy, details)
    pass


# ===========================================================================
# TASK 3 — Visualising Embedding Space
# ===========================================================================

# Word groups used for visualisation — already defined for you.
WORD_GROUPS: dict[str, list[str]] = {
    "animals":     ["cat", "dog", "horse", "lion", "tiger", "eagle", "shark",
                    "rabbit", "wolf", "bear"],
    "countries":   ["france", "germany", "italy", "spain", "japan", "china",
                    "brazil", "india", "canada", "mexico"],
    "professions": ["doctor", "nurse", "teacher", "engineer", "lawyer",
                    "scientist", "artist", "chef", "pilot", "programmer"],
    "emotions":    ["happy", "sad", "angry", "fearful", "surprised",
                    "disgusted", "joyful", "anxious", "calm", "excited"],
    "royalty":     ["king", "queen", "prince", "princess", "emperor",
                    "duchess", "lord", "knight", "throne", "crown"],
}


def collect_word_vectors(
    groups: dict[str, list[str]],
    kv: KeyedVectors,
) -> tuple[list[str], list[str], np.ndarray]:
    """
    Gather all words from `groups` that exist in kv.

    Returns
    -------
    words       : list[str]       — kept words
    group_labels: list[str]       — group name per word (same order)
    matrix      : np.ndarray      — shape (n_words, emb_dim)
    """
    words, group_labels, vectors = [], [], []
    for group_name, word_list in groups.items():
        for word in word_list:
            if word in kv:
                words.append(word)
                group_labels.append(group_name)
                vectors.append(kv[word])
    matrix = np.stack(vectors, axis=0)  # (n_words, emb_dim)
    return words, group_labels, matrix


def reduce_with_pca(
    matrix: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce `matrix` to `n_components` dimensions using PCA.

    Steps
    -----
    1. Import PCA from sklearn.decomposition.
    2. Fit on matrix and transform it.
    3. Return the result.

    Parameters
    ----------
    matrix       : np.ndarray  shape (n, d)
    n_components : int

    Returns
    -------
    np.ndarray  shape (n, n_components)
    """
    # TODO: apply PCA and return reduced matrix
    pass


def reduce_with_tsne(
    matrix: np.ndarray,
    n_components: int = 2,
    perplexity: float = 15.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce `matrix` to `n_components` dimensions using t-SNE.

    Tip: run PCA first (to ~50 dims) to speed up t-SNE on high-dimensional data.

    Steps
    -----
    1. If matrix.shape[1] > 50, apply PCA to reduce to 50 dims first.
    2. Apply TSNE from sklearn.manifold.
    3. Return the 2-D result.

    Parameters
    ----------
    matrix       : np.ndarray  shape (n, d)
    n_components : int
    perplexity   : float  (t-SNE hyperparameter, try 5–50)
    random_state : int

    Returns
    -------
    np.ndarray  shape (n, n_components)
    """
    # TODO: optionally reduce dims with PCA, then apply TSNE, return result
    pass


def plot_embeddings(
    coords: np.ndarray,
    words: list[str],
    group_labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    """
    Scatter-plot 2-D embeddings, colour-coded by group, with word labels.

    Steps
    -----
    1. Import matplotlib.pyplot as plt.
    2. Assign a unique colour to each distinct group (use a colour cycle).
    3. Plot each point; annotate it with its word.
    4. Add a legend showing group names.
    5. Set the title and save the figure to output_path.

    Parameters
    ----------
    coords      : np.ndarray  shape (n, 2)
    words       : list[str]
    group_labels: list[str]   — one entry per word
    title       : str
    output_path  : Path
    """
    # TODO: build the scatter plot and save to output_path
    pass


# ===========================================================================
# TASK 4 — Sentence Representation via Averaging
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 5 – Mean document embedding
# ---------------------------------------------------------------------------

def embed_document(text: str, kv: KeyedVectors) -> np.ndarray:
    """
    Convert a text to a fixed-size vector by averaging in-vocabulary tokens.

    Algorithm
    ---------
    1. Tokenise with simple_tokenize.
    2. Keep only tokens present in kv.
    3. Return the mean of their vectors.
    4. If no token is found, return a zero vector of shape (kv.vector_size,).

    Parameters
    ----------
    text : str
    kv   : KeyedVectors

    Returns
    -------
    np.ndarray  shape (kv.vector_size,)
    """
    # TODO: collect in-vocab vectors, return mean (or zeros if none found)
    pass


# ---------------------------------------------------------------------------
# Step 7 – Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_embeddings(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[int],
    y_test: list[int],
    label: str,
) -> None:
    """Train LogisticRegression on embeddings and print a classification report."""
    # TODO: fit LogisticRegression(max_iter=1000, random_state=42)
    # TODO: predict on X_test
    # TODO: compute macro F1 and print section header + classification_report
    pass


# ===========================================================================
# TASK 5 — Bias Detection
# ===========================================================================

# Word sets used as gender poles — already defined for you.
MALE_WORDS: list[str] = [
    "man", "he", "his", "him", "male", "boy", "father",
    "brother", "son", "husband", "uncle", "grandfather",
]
FEMALE_WORDS: list[str] = [
    "woman", "she", "her", "hers", "female", "girl", "mother",
    "sister", "daughter", "wife", "aunt", "grandmother",
]
PROFESSIONS: list[str] = [
    "doctor", "nurse", "engineer", "teacher", "lawyer", "scientist",
    "artist", "chef", "pilot", "programmer", "secretary", "president",
    "manager", "receptionist", "plumber", "librarian",
]


def compute_gender_bias_score(
    word: str,
    kv: KeyedVectors,
    male_words: list[str],
    female_words: list[str],
) -> float:
    """
    Compute a scalar gender-bias score for `word`.

    Method
    ------
    1. Compute the mean cosine similarity of `word` to all MALE_WORDS that
       exist in kv  →  male_sim
    2. Compute the mean cosine similarity of `word` to all FEMALE_WORDS that
       exist in kv  →  female_sim
    3. Return bias = male_sim - female_sim

    Interpretation
    --------------
     positive → more associated with male pole
     zero     → neutral
     negative → more associated with female pole

    Skip gender words that are not in kv.
    Return 0.0 if either set produces no valid words.

    Parameters
    ----------
    word        : the target profession / concept
    kv          : KeyedVectors
    male_words  : list of male-pole words
    female_words: list of female-pole words

    Returns
    -------
    float
    """
    if word not in kv:
        return 0.0

    # TODO: compute mean cosine similarity to each gender pole
    # TODO: return male_sim - female_sim
    pass


def report_profession_bias(
    professions: list[str],
    kv: KeyedVectors,
    male_words: list[str],
    female_words: list[str],
) -> list[tuple[str, float]]:
    """
    Compute bias scores for every profession and return them sorted from most
    male-associated to most female-associated.

    Parameters
    ----------
    professions : list of target words
    kv          : KeyedVectors
    male_words  : list of male-pole words
    female_words: list of female-pole words

    Returns
    -------
    list[tuple[str, float]]  — (profession, bias_score) sorted descending
    """
    # TODO: call compute_gender_bias_score for each profession
    # TODO: sort by score descending and return
    pass


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # ---------------------------------------------------------------
    # Load embeddings  (shared by all tasks)
    # ---------------------------------------------------------------
    print("Loading GloVe embeddings (may take a moment on first run)...")
    kv = load_embeddings()
    print(f"  Vocabulary size : {len(kv.index_to_key):,}")
    print(f"  Embedding dim   : {kv.vector_size}")

    # ===================================================================
    # TASK 1 – Word Similarity & Nearest Neighbours
    # ===================================================================
    print("\n" + "=" * 60)
    print("  TASK 1 — Word Similarity & Nearest Neighbours")
    print("=" * 60)

    # 1a. Basic nearest neighbours
    for query in ["king", "doctor", "bank", "python"]:
        neighbours = most_similar_words(query, kv, top_n=5)
        print(f"\nTop-5 neighbours of '{query}':")
        for word, sim in neighbours:
            print(f"  {word:<20s}  {sim:.4f}")

    # 1b. Polysemy — "bank"
    # The embedding for "bank" blends financial and river-bank senses.
    # Does averaging "river" + "bank" shift the neighbours toward the waterway sense?
    print("\n--- Polysemy: 'bank' vs average(['river', 'bank']) ---")
    avg_vec = average_query(["river", "bank"], kv)
    river_bank_neighbours = most_similar_to_vector(
        avg_vec, kv, exclude={"river", "bank"}, top_n=5
    )
    print("Top-5 neighbours of avg(['river', 'bank']):")
    for word, sim in river_bank_neighbours:
        print(f"  {word:<20s}  {sim:.4f}")

    # ===================================================================
    # TASK 2 – Word Analogy
    # ===================================================================
    print("\n" + "=" * 60)
    print("  TASK 2 — Word Analogy")
    print("=" * 60)

    # 2a. Classic examples
    classic = [
        ("man",   "king",   "woman"),
        ("paris", "france", "berlin"),
        ("walk",  "walked", "run"),
        ("slow",  "slower", "fast"),
    ]
    print("\nClassic analogies (a : b :: c : ?):")
    for a, b, c in classic:
        results = solve_analogy(a, b, c, kv, top_n=3)
        answers = ", ".join(f"{w} ({s:.3f})" for w, s in results)
        print(f"  {a} : {b}  ::  {c} : ?   →  {answers}")

    # 2b. Benchmark evaluation
    print("\nBenchmark evaluation:")
    accuracy, details = evaluate_analogies(ANALOGY_BENCHMARK, kv)
    for d in details:
        tick = "✓" if d["correct"] else "✗"
        print(f"  {tick} {d['analogy']:<32s}  expected={d['expected']:<12s}  "
              f"got={d['predicted']}")
    print(f"\n  Accuracy: {accuracy:.1%}  ({sum(d['correct'] for d in details)}"
          f"/{len(details)} correct)")

    # ===================================================================
    # TASK 3 – Visualising Embedding Space
    # ===================================================================
    print("\n" + "=" * 60)
    print("  TASK 3 — Visualising Embedding Space")
    print("=" * 60)

    words, group_labels, matrix = collect_word_vectors(WORD_GROUPS, kv)
    print(f"  Collected {len(words)} words across {len(WORD_GROUPS)} groups")

    # PCA projection
    print("  Projecting with PCA...")
    coords_pca = reduce_with_pca(matrix, n_components=2)
    plot_embeddings(
        coords_pca, words, group_labels,
        title="Word Embeddings — PCA projection",
        output_path=VIZ_PATH.parent / "embedding_pca.png",
    )
    print(f"  PCA plot saved → {VIZ_PATH.parent / 'embedding_pca.png'}")

    # t-SNE projection
    print("  Projecting with t-SNE (this may take ~30 s)...")
    coords_tsne = reduce_with_tsne(matrix, n_components=2, perplexity=15)
    plot_embeddings(
        coords_tsne, words, group_labels,
        title="Word Embeddings — t-SNE projection",
        output_path=VIZ_PATH.parent / "embedding_tsne.png",
    )
    print(f"  t-SNE plot saved → {VIZ_PATH.parent / 'embedding_tsne.png'}")

    # Reflection — answer these in your report:
    # Q1: Do words from the same semantic group cluster together in PCA? In t-SNE?
    # Q2: Are antonyms (e.g. happy/sad) close together or far apart? Why?

    # ===================================================================
    # TASK 4 – Sentence Representation via Averaging
    # ===================================================================
    print("\n" + "=" * 60)
    print("  TASK 4 — Sentence Embedding & Classification")
    print("=" * 60)

    texts, labels = load_spam_dataset(DATA_PATH)
    print(f"  Samples: {len(texts)}  |  Spam: {sum(labels)}  Ham: {len(labels)-sum(labels)}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.5, random_state=42, stratify=labels
    )

    # Mean embedding
    print("\nBuilding mean-embedding features...")
    X_train_mean = np.array([embed_document(t, kv) for t in X_train_raw])
    X_test_mean  = np.array([embed_document(t, kv) for t in X_test_raw])
    evaluate_embeddings(X_train_mean, X_test_mean, y_train, y_test,
                        "Mean Word Embeddings")

    # Basic TF-IDF text features (sklearn implementation)
    print("\nBuilding basic TF-IDF features (sklearn)...")
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenize,
        lowercase=False,
        token_pattern=None,
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
    evaluate_embeddings(X_train_tfidf, X_test_tfidf, y_train, y_test,
                        "Basic TF-IDF (sklearn)")

    # Reflection — answer these in your report:
    # Q1: Does basic TF-IDF improve over mean word embeddings? Why or why not?
    # Q2: What types of sentences would averaging fail on?
    #     (Hint: think about negation — "not happy" vs "happy")
    # Q3: How would you represent "The bank was on the river bank" correctly?

    # ===================================================================
    # TASK 5 – Bias Detection
    # ===================================================================
    print("\n" + "=" * 60)
    print("  TASK 5 — Gender Bias in Embeddings")
    print("=" * 60)
    print("  Bias score = mean_sim(word, male_words) - mean_sim(word, female_words)")
    print("  Positive → male-skewed  |  Negative → female-skewed\n")

    ranked = report_profession_bias(PROFESSIONS, kv, MALE_WORDS, FEMALE_WORDS)
    print(f"  {'Profession':<16}  {'Bias score':>10}  Direction")
    print("  " + "-" * 40)
    for profession, score in ranked:
        direction = "→ male" if score > 0.01 else ("→ female" if score < -0.01 else "≈ neutral")
        bar_len   = int(abs(score) * 80)
        bar       = ("+" if score > 0 else "-") * bar_len
        print(f"  {profession:<16}  {score:>10.4f}  {direction}  {bar}")

    # Reflection — answer these in your report:
    # Q1: Which professions carry the strongest gender associations?
    # Q2: Do you find any surprising results? What might explain them?
    # Q3: These embeddings were trained on Wikipedia + Gigaword (news text from 2014).
    #     How might the training corpus affect the biases you observe?
    # Q4: Is it always harmful to have gendered word associations?
    #     When could they be useful, and when are they problematic?


if __name__ == "__main__":
    main()
