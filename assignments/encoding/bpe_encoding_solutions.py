"""
BPE Encoding Assignment (Solutions)
===================================
"""

from collections import defaultdict


WordSymbols = tuple[str, ...]
Pair = tuple[str, str]


def build_initial_vocab(corpus: list[str]) -> dict[WordSymbols, int]:
    word_counts: dict[str, int] = defaultdict(int)
    for sentence in corpus:
        for word in sentence.strip().split():
            if word:
                word_counts[word] += 1

    vocab: dict[WordSymbols, int] = {}
    for word, freq in word_counts.items():
        symbols: WordSymbols = tuple(list(word) + ["</w>"])
        vocab[symbols] = freq
    return vocab


def get_pair_counts(vocab: dict[WordSymbols, int]) -> dict[Pair, int]:
    pair_counts: dict[Pair, int] = defaultdict(int)
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pair_counts[(a, b)] += freq
    return dict(pair_counts)


def merge_pair(pair: Pair, vocab: dict[WordSymbols, int]) -> dict[WordSymbols, int]:
    merged_token = pair[0] + pair[1]
    new_vocab: dict[WordSymbols, int] = {}

    for symbols, freq in vocab.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1

        new_word = tuple(new_symbols)
        new_vocab[new_word] = new_vocab.get(new_word, 0) + freq

    return new_vocab


def train_bpe(corpus: list[str], num_merges: int) -> tuple[list[Pair], dict[WordSymbols, int]]:
    vocab = build_initial_vocab(corpus)
    merges: list[Pair] = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break

        # Deterministic tie-break: highest frequency, then lexicographically smallest pair.
        best_pair = sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        merges.append(best_pair)
        vocab = merge_pair(best_pair, vocab)

    return merges, vocab


def _apply_single_merge(symbols: list[str], merge_rule: Pair) -> list[str]:
    merged = merge_rule[0] + merge_rule[1]
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == merge_rule:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def encode_word(word: str, merges: list[Pair]) -> list[str]:
    symbols: list[str] = list(word) + ["</w>"]
    for merge_rule in merges:
        symbols = _apply_single_merge(symbols, merge_rule)

    if symbols and symbols[-1] == "</w>":
        symbols = symbols[:-1]
    elif symbols and symbols[-1].endswith("</w>"):
        symbols[-1] = symbols[-1].replace("</w>", "")
    return symbols


def encode_text(text: str, merges: list[Pair]) -> list[str]:
    encoded: list[str] = []
    for word in text.strip().split():
        encoded.extend(encode_word(word, merges))
    return encoded


def main() -> None:
    corpus = [
        "low lower lowest",
        "newer wider",
        "low low lower",
    ]
    num_merges = 10

    merges, final_vocab = train_bpe(corpus, num_merges=num_merges)

    print("Learned merges:")
    for i, p in enumerate(merges, start=1):
        print(f"{i:02d}. {p[0]} + {p[1]}")

    print("\nFinal vocab entries:")
    for symbols, freq in sorted(final_vocab.items(), key=lambda x: (-x[1], x[0])):
        print(f"{symbols} -> {freq}")

    test_text = "lowest newer low"
    encoded = encode_text(test_text, merges)
    print("\nTest text:", test_text)
    print("Encoded tokens:", encoded)


if __name__ == "__main__":
    main()
