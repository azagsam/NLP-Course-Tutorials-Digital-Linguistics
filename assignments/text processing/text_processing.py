"""
Alternative Text Processing Assignment (Student Version)
=======================================================

Complete the TODO sections to build a full NLTK vs Stanza comparison pipeline.
This file is intentionally scaffolded for students.
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

        # TODO: Sentence tokenization using NLTK.
        sentences = []

        # TODO: Word tokenization using NLTK.
        tokens = []

        # TODO: POS tagging using NLTK.
        pos_tags_result = []

        # TODO: Lemmatize each token using WordNetLemmatizer.
        lemmas = []

        # TODO: Named Entity Recognition with ne_chunk over POS tags.
        # Store entities as tuples: (entity_text, entity_label).
        entities: list[tuple[str, str]] = []

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
        # TODO 5.1: Initialize a Stanza pipeline for English with correct arguments
        pass

    def process(self, text: str) -> PipelineResult:
        t0 = time.perf_counter()

        # TODO: Run Stanza pipeline on text.
        doc = None

        # TODO: Extract sentence texts.
        sentences = []

        # TODO: Extract tokens (word text).
        tokens = []

        # TODO: Extract POS tags as (word, tag).
        pos_tags_result = []

        # TODO: Extract lemmas.
        lemmas = []

        # TODO: Extract named entities as (entity_text, entity_type).
        entities = []

        return PipelineResult(
            sentences=sentences,
            tokens=tokens,
            pos_tags=pos_tags_result,
            lemmas=lemmas,
            entities=entities,
            elapsed_s=time.perf_counter() - t0,
        )


def compare_counts(nltk_res: PipelineResult, stanza_res: PipelineResult) -> dict[str, int]:
    # TODO: Return a dictionary with these keys:
    # sentences_nltk, sentences_stanza, tokens_nltk, tokens_stanza,
    # entities_nltk, entities_stanza
    return {
        "sentences_nltk": 0,
        "sentences_stanza": 0,
        "tokens_nltk": 0,
        "tokens_stanza": 0,
        "entities_nltk": 0,
        "entities_stanza": 0,
    }


def visualize_token_counts(rows: list[dict[str, str]], output_path: Path) -> None:
    # TODO: Build labels: S1, S2, ... based on number of rows.
    labels = []

    # TODO: Read token counts from rows and convert to int.
    nltk_counts = []
    stanza_counts = []

    # TODO: Create a grouped bar chart (NLTK vs Stanza) and save it.
    # Requirements:
    # - X-axis: sample labels
    # - Y-axis: number of tokens
    # - title: "Token Count Comparison: NLTK vs Stanza"
    # - legend enabled
    # - save to output_path with dpi=150
    pass


def write_report(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Text Processing Comparison Report",
        "",
        "Fill in the analysis prompts under each sample after running the script.",
        "",
    ]

    for i, row in enumerate(rows, start=1):
        # TODO: Add a markdown block for each sample containing:
        # - sample heading
        # - source text
        # - timings
        # - token/entity counts
        # - quick inspection previews
        # - analysis prompts
        # Tip: use lines.extend([...])
        pass

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

        # TODO: Append a dictionary to report_rows containing:
        # text, nltk_time, stanza_time,
        # tokens_nltk, tokens_stanza,
        # entities_nltk, entities_stanza,
        # nltk_tokens_preview, stanza_tokens_preview,
        # nltk_pos_preview, stanza_pos_preview,
        # nltk_entities, stanza_entities
        # Use previews: first 12 tokens and first 8 POS tags.
        pass

    report_path = Path("assignments/text_processing_alternative_report.md")
    plot_path = Path("assignments/token_count_comparison.png")

    # TODO: Call write_report and visualize_token_counts using report_rows.

    print("Student assignment executed.")
    print("Complete all TODO sections to generate full output files.")


if __name__ == "__main__":
    run()
