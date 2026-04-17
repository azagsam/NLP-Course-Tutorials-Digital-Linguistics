"""
Alternative Text Processing Assignment
====================================

This script provides an alternative assignment format to `text_processing.py`.
It compares NLTK and Stanza pipelines and generates a structured markdown report.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
from nltk import ne_chunk, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

import stanza


TEXTS = [
    "Wait... did Dr. J. Smith (U.C. Berkeley) really say 'NLP is easy' at 3:30 p.m., or was it sarcasm?!",
    "Email me at first.last+nlp@uni-example.edu ASAP - unless you've already sent it via https://tinyurl.com/nlp-demo.",
    "The startup's Q4 revenue was $1.2M-ish (not audited), yet users wrote: 'app crashes on iOS17/Android14 :('",
    "I re-read the note: \"Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.\" Still parsing it...",
]


@dataclass
class PipelineResult:
    sentences: list[str]
    tokens: list[str]
    pos_tags: list[tuple[str, str]]
    lemmas: list[str]
    entities: list[tuple[str, str]]
    elapsed_s: float


class NLTKPipeline:
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()

    def process(self, text: str) -> PipelineResult:
        t0 = time.perf_counter()

        sentences = sent_tokenize(text)
        tokens = word_tokenize(text)
        pos_tags_result = pos_tag(tokens)
        lemmas = [self.lemmatizer.lemmatize(tok) for tok in tokens]

        chunked = ne_chunk(pos_tags_result)
        entities: list[tuple[str, str]] = []
        for node in chunked:
            if hasattr(node, "label"):
                ent_text = " ".join(word for word, _ in node)
                entities.append((ent_text, node.label()))

        return PipelineResult(
            sentences=sentences,
            tokens=tokens,
            pos_tags=pos_tags_result,
            lemmas=lemmas,
            entities=entities,
            elapsed_s=time.perf_counter() - t0,
        )


class StanzaPipeline:
    def __init__(self) -> None:
        self.nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,ner", verbose=False)

    def process(self, text: str) -> PipelineResult:
        t0 = time.perf_counter()
        doc = self.nlp(text)

        sentences = [sent.text for sent in doc.sentences]
        tokens = [w.text for sent in doc.sentences for w in sent.words]
        pos_tags_result = [(w.text, w.xpos) for sent in doc.sentences for w in sent.words]
        lemmas = [w.lemma for sent in doc.sentences for w in sent.words]
        entities = [(ent.text, ent.type) for ent in doc.ents]

        return PipelineResult(
            sentences=sentences,
            tokens=tokens,
            pos_tags=pos_tags_result,
            lemmas=lemmas,
            entities=entities,
            elapsed_s=time.perf_counter() - t0,
        )


def compare_counts(nltk_res: PipelineResult, stanza_res: PipelineResult) -> dict[str, int]:
    return {
        "sentences_nltk": len(nltk_res.sentences),
        "sentences_stanza": len(stanza_res.sentences),
        "tokens_nltk": len(nltk_res.tokens),
        "tokens_stanza": len(stanza_res.tokens),
        "entities_nltk": len(nltk_res.entities),
        "entities_stanza": len(stanza_res.entities),
    }


def visualize_token_counts(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = [f"S{i}" for i in range(1, len(rows) + 1)]
    nltk_counts = [int(row["tokens_nltk"]) for row in rows]
    stanza_counts = [int(row["tokens_stanza"]) for row in rows]

    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width / 2 for i in x], nltk_counts, width=width, label="NLTK")
    plt.bar([i + width / 2 for i in x], stanza_counts, width=width, label="Stanza")
    plt.xticks(x, labels)
    plt.xlabel("Sample")
    plt.ylabel("Number of tokens")
    plt.title("Token Count Comparison: NLTK vs Stanza")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_report(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Text Processing Comparison Report",
        "",
        "Fill in the analysis prompts under each sample after running the script.",
        "",
    ]

    for i, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Sample {i}",
                f"- Text: `{row['text']}`",
                f"- NLTK time (s): {row['nltk_time']}",
                f"- Stanza time (s): {row['stanza_time']}",
                f"- Token counts: NLTK={row['tokens_nltk']}, Stanza={row['tokens_stanza']}",
                f"- Entity counts: NLTK={row['entities_nltk']}, Stanza={row['entities_stanza']}",
                "",
                "### Quick Inspection",
                f"- NLTK tokens (first 12): `{row['nltk_tokens_preview']}`",
                f"- Stanza tokens (first 12): `{row['stanza_tokens_preview']}`",
                f"- NLTK POS (first 8): `{row['nltk_pos_preview']}`",
                f"- Stanza POS (first 8): `{row['stanza_pos_preview']}`",
                f"- NLTK entities: `{row['nltk_entities']}`",
                f"- Stanza entities: `{row['stanza_entities']}`",
                "",
                "### Student Analysis",
                "- Tokenization differences:",
                "- POS-tagging differences:",
                "- Lemmatization differences:",
                "- NER differences:",
                "- Which pipeline is preferable for this sample and why:",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_nltk_resources() -> None:
    required = [
        "punkt",
        "averaged_perceptron_tagger",
        "wordnet",
        "maxent_ne_chunker",
        "words",
    ]
    for item in required:
        nltk.download(item, quiet=True)


def run() -> None:
    ensure_nltk_resources()
    stanza.download("en", verbose=False)

    nltk_pipe = NLTKPipeline()
    stanza_pipe = StanzaPipeline()

    report_rows: list[dict[str, str]] = []

    for text in TEXTS:
        nltk_res = nltk_pipe.process(text)
        stanza_res = stanza_pipe.process(text)
        counts = compare_counts(nltk_res, stanza_res)

        report_rows.append(
            {
                "text": text,
                "nltk_time": f"{nltk_res.elapsed_s:.4f}",
                "stanza_time": f"{stanza_res.elapsed_s:.4f}",
                "tokens_nltk": str(counts["tokens_nltk"]),
                "tokens_stanza": str(counts["tokens_stanza"]),
                "entities_nltk": str(counts["entities_nltk"]),
                "entities_stanza": str(counts["entities_stanza"]),
                "nltk_tokens_preview": str(nltk_res.tokens[:12]),
                "stanza_tokens_preview": str(stanza_res.tokens[:12]),
                "nltk_pos_preview": str(nltk_res.pos_tags[:8]),
                "stanza_pos_preview": str(stanza_res.pos_tags[:8]),
                "nltk_entities": str(nltk_res.entities),
                "stanza_entities": str(stanza_res.entities),
            }
        )

    report_path = Path("assignments/text_processing_alternative_report.md")
    plot_path = Path("assignments/token_count_comparison.png")

    write_report(report_rows, report_path)
    visualize_token_counts(report_rows, plot_path)

    print("Alternative assignment completed.")
    print(f"Report written to: {report_path}")
    print(f"Token chart written to: {plot_path}")
    print("Next: Open the report and fill in each 'Student Analysis' section.")


if __name__ == "__main__":
    run()
