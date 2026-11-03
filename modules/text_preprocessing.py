"""Unicode-safe text normalization and sentence-conscious token chunking."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

_WHITESPACE = re.compile(r"[^\S\r\n]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving punctuation and words."""
    normalized = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def split_sentences(text: str) -> list[str]:
    """Split common prose without depending on a downloaded NLP model."""
    return [part.strip() for part in _SENTENCE.split(clean_text(text)) if part.strip()]


def chunk_text(
    text: str,
    tokenizer: object | None = None,
    max_tokens: int = 900,
    max_chars_fallback: int = 3600,
) -> list[str]:
    """Create tokenizer-aware chunks, breaking only inside long sentences if needed."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    def token_count(value: str) -> int:
        if tokenizer is not None:
            encode = getattr(tokenizer, "encode", None)
            if encode:
                return len(encode(value, add_special_tokens=False))
        return max(1, len(value) // 4)

    def emit_long(sentence: str) -> list[str]:
        words = sentence.split()
        pieces: list[str] = []
        piece: list[str] = []
        for word in words:
            candidate = " ".join(piece + [word])
            if piece and (
                token_count(candidate) > max_tokens
                or (tokenizer is None and len(candidate) > max_chars_fallback)
            ):
                pieces.append(" ".join(piece))
                piece = [word]
            else:
                piece.append(word)
        if piece:
            pieces.append(" ".join(piece))
        return pieces

    for sentence in sentences:
        if token_count(sentence) > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_size = [], 0
            chunks.extend(emit_long(sentence))
            continue
        proposed = " ".join(current + [sentence])
        proposed_size = token_count(proposed)
        if current and proposed_size > max_tokens:
            chunks.append(" ".join(current))
            current, current_size = [sentence], token_count(sentence)
        else:
            current.append(sentence)
            current_size = proposed_size
    if current:
        chunks.append(" ".join(current))
    return chunks
