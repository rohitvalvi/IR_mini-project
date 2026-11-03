"""Evaluation and simple extractive baseline utilities."""

from __future__ import annotations

import math
import re
from collections import Counter

from .text_preprocessing import split_sentences


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def compression_percentage(original: str, summary: str) -> float:
    original_count = word_count(original)
    if original_count == 0:
        return 0.0
    return max(0.0, (1 - word_count(summary) / original_count) * 100)


def rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
    """Return ROUGE-1/2/L F1; dependency is optional until this function is used."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("Install rouge-score to calculate ROUGE metrics.") from exc
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {name: round(value.fmeasure, 4) for name, value in scores.items()}


def tfidf_baseline(text: str, sentence_count: int = 3) -> str:
    """Return a deterministic TF-IDF-like extractive baseline without sklearn."""
    sentences = split_sentences(text)
    if len(sentences) <= sentence_count:
        return " ".join(sentences)
    tokenized = [re.findall(r"\b\w+\b", sentence.lower()) for sentence in sentences]
    document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
    scored: list[tuple[float, int, str]] = []
    total = len(sentences)
    for index, tokens in enumerate(tokenized):
        if not tokens:
            continue
        tf = Counter(tokens)
        score = sum(
            (count / len(tokens)) * math.log((1 + total) / (1 + document_frequency[token]))
            for token, count in tf.items()
        )
        scored.append((score, index, sentences[index]))
    chosen = sorted(scored, reverse=True)[:sentence_count]
    return " ".join(sentence for _, _, sentence in sorted(chosen, key=lambda item: item[1]))
