"""
Word Embeddings Assignment (Solutions Version)
===============================================

Five tasks covering word similarity, analogy solving, visualisation,
sentence averaging, and bias detection with GloVe-100 embeddings.

Run from the repo root:
    python assignments/word2vec/word2vec_embeddings_solutions.py
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("assignments/data/spam.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_spam_dataset(path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    texts  = df["text"].astype(str).tolist()
    labels = (df["label"].str.lower() == "spam").astype(int).tolist()
    return texts, labels


def simple_tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t.isalpha()]


# ===========================================================================
# Step 1 – Load pre-trained embeddings
# ===========================================================================

def load_embeddings(model_name: str = "glove-wiki-gigaword-100") -> KeyedVectors:
    print(f"  Downloading / loading '{model_name}'…")
    return api.load(model_name)


# ===========================================================================
# TASK 1 — Word Similarity & Nearest Neighbours
# ===========================================================================

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def most_similar_words(
    word: str,
    kv: KeyedVectors,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    if word not in kv:
        raise KeyError(f"'{word}' not in vocabulary")

    query_vec = kv[word]
    scores: list[tuple[str, float]] = [
        (w, cosine_similarity(query_vec, kv[w]))
        for w in kv.index_to_key
        if w != word
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def average_query(words: list[str], kv: KeyedVectors) -> np.ndarray:
    vecs = [kv[w] for w in words if w in kv]
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def most_similar_to_vector(
    query_vec: np.ndarray,
    kv: KeyedVectors,
    exclude: set[str] | None = None,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    exclude = exclude or set()
    scores: list[tuple[str, float]] = [
        (w, cosine_similarity(query_vec, kv[w]))
        for w in kv.index_to_key
        if w not in exclude
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ===========================================================================
# TASK 2 — Word Analogy
# ===========================================================================

def solve_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    kv: KeyedVectors,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    for w in (word_a, word_b, word_c):
        if w not in kv:
            raise KeyError(f"'{w}' not in vocabulary")

    query_vector = kv[word_b] - kv[word_a] + kv[word_c]
    exclude      = {word_a, word_b, word_c}

    scores: list[tuple[str, float]] = [
        (w, cosine_similarity(query_vector, kv[w]))
        for w in kv.index_to_key
        if w not in exclude
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


ANALOGY_BENCHMARK: list[tuple[str, str, str, str]] = [
    ("paris",    "france",   "rome",    "italy"),
    ("paris",    "france",   "berlin",  "germany"),
    ("paris",    "france",   "madrid",  "spain"),
    ("man",      "king",     "woman",   "queen"),
    ("man",      "actor",    "woman",   "actress"),
    ("man",      "father",   "woman",   "mother"),
    ("big",      "bigger",   "small",   "smaller"),
    ("good",     "best",     "bad",     "worst"),
    ("fast",     "faster",   "slow",    "slower"),
    ("walk",     "walked",   "run",     "ran"),
    ("swim",     "swam",     "fly",     "flew"),
    ("go",       "went",     "come",    "came"),
]


def evaluate_analogies(
    benchmark: list[tuple[str, str, str, str]],
    kv: KeyedVectors,
) -> tuple[float, list[dict]]:
    details: list[dict] = []
    n_correct = 0

    for word_a, word_b, word_c, expected in benchmark:
        analogy_str = f"{word_a}:{word_b} :: {word_c}:?"
        try:
            results   = solve_analogy(word_a, word_b, word_c, kv, top_n=1)
            predicted = results[0][0] if results else "<none>"
        except KeyError:
            predicted = "<OOV>"

        correct = predicted == expected
        n_correct += int(correct)
        details.append({
            "analogy":   analogy_str,
            "expected":  expected,
            "predicted": predicted,
            "correct":   correct,
        })

    return n_correct / len(benchmark), details


# ===========================================================================
# TASK 3 — Visualising Embedding Space
# ===========================================================================

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
    words, group_labels, vectors = [], [], []
    for group_name, word_list in groups.items():
        for word in word_list:
            if word in kv:
                words.append(word)
                group_labels.append(group_name)
                vectors.append(kv[word])
    return words, group_labels, np.stack(vectors, axis=0)


def reduce_with_pca(
    matrix: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    pca = PCA(n_components=n_components)
    return pca.fit_transform(matrix)


def reduce_with_tsne(
    matrix: np.ndarray,
    n_components: int = 2,
    perplexity: float = 15.0,
    random_state: int = 42,
) -> np.ndarray:
    # Pre-reduce with PCA to speed up t-SNE on high-dimensional input
    if matrix.shape[1] > 50:
        matrix = PCA(n_components=50, random_state=random_state).fit_transform(matrix)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
    )
    return tsne.fit_transform(matrix)


def plot_embeddings(
    coords: np.ndarray,
    words: list[str],
    group_labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    unique_groups  = list(dict.fromkeys(group_labels))   # preserve insertion order
    colour_map     = cm.get_cmap("tab10", len(unique_groups))
    group_to_colour = {g: colour_map(i) for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each word
    for i, (word, group) in enumerate(zip(words, group_labels)):
        colour = group_to_colour[group]
        ax.scatter(coords[i, 0], coords[i, 1], color=colour, s=40, alpha=0.8)
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            fontsize=8,
            alpha=0.9,
            color=colour,
        )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=group_to_colour[g], markersize=8, label=g)
        for g in unique_groups
    ]
    ax.legend(handles=handles, title="Group", loc="best", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ===========================================================================
# TASK 4 — Sentence Representation via Averaging
# ===========================================================================

def embed_document(text: str, kv: KeyedVectors) -> np.ndarray:
    vectors = [kv[t] for t in simple_tokenize(text) if t in kv]
    if not vectors:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def evaluate_embeddings(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[int],
    y_test: list[int],
    label: str,
) -> None:
    clf  = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    f1   = f1_score(y_test, pred, average="macro", zero_division=0)
    sep  = "=" * 52
    print(f"\n{sep}\n  {label}\n{sep}")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=4))
    print(f"  Macro F1: {f1:.4f}")


# ===========================================================================
# TASK 5 — Bias Detection
# ===========================================================================

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
    if word not in kv:
        return 0.0

    word_vec   = kv[word]
    male_sims  = [cosine_similarity(word_vec, kv[w]) for w in male_words   if w in kv]
    female_sims= [cosine_similarity(word_vec, kv[w]) for w in female_words if w in kv]

    if not male_sims or not female_sims:
        return 0.0
    return float(np.mean(male_sims) - np.mean(female_sims))


def report_profession_bias(
    professions: list[str],
    kv: KeyedVectors,
    male_words: list[str],
    female_words: list[str],
) -> list[tuple[str, float]]:
    scored = [
        (p, compute_gender_bias_score(p, kv, male_words, female_words))
        for p in professions
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
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

    for query in ["king", "doctor", "bank", "python"]:
        neighbours = most_similar_words(query, kv, top_n=5)
        print(f"\nTop-5 neighbours of '{query}':")
        for word, sim in neighbours:
            print(f"  {word:<20s}  {sim:.4f}")

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

    print("  Projecting with PCA...")
    coords_pca = reduce_with_pca(matrix, n_components=2)
    pca_path   = Path("assignments/word2vec/embedding_pca.png")
    plot_embeddings(coords_pca, words, group_labels,
                    title="Word Embeddings — PCA projection",
                    output_path=pca_path)
    print(f"  PCA plot saved → {pca_path}")

    print("  Projecting with t-SNE (this may take ~30 s)...")
    coords_tsne = reduce_with_tsne(matrix, n_components=2, perplexity=15)
    tsne_path   = Path("assignments/word2vec/embedding_tsne.png")
    plot_embeddings(coords_tsne, words, group_labels,
                    title="Word Embeddings — t-SNE projection",
                    output_path=tsne_path)
    print(f"  t-SNE plot saved → {tsne_path}")

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

    print("\nBuilding mean-embedding features...")
    X_train_mean = np.array([embed_document(t, kv) for t in X_train_raw])
    X_test_mean  = np.array([embed_document(t, kv) for t in X_test_raw])
    evaluate_embeddings(X_train_mean, X_test_mean, y_train, y_test,
                        "Mean Word Embeddings")

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


if __name__ == "__main__":
    main()
