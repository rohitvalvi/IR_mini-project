"""Lazy, Apple-Silicon-friendly abstractive summarization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from .text_preprocessing import chunk_text

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


class SummaryLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    DETAILED = "detailed"


class SummarizationError(RuntimeError):
    """Raised for expected model and input failures."""


@dataclass(frozen=True)
class GenerationConfig:
    min_length: int
    max_length: int


_CONFIGS = {
    SummaryLength.SHORT: GenerationConfig(35, 90),
    SummaryLength.MEDIUM: GenerationConfig(60, 150),
    SummaryLength.DETAILED: GenerationConfig(100, 220),
}


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_summarizer() -> "DocumentSummarizer":
    """Construct and cache one model/tokenizer pair per Python process."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        import torch

        device = _device()
        model.to(torch.device(device))
        model.eval()
        return DocumentSummarizer(tokenizer, model, device)
    except Exception as exc:
        raise SummarizationError(
            "The summarization model could not be loaded. Ensure Transformers/PyTorch "
            "are installed and that the first run has internet access."
        ) from exc


class DocumentSummarizer:
    def __init__(self, tokenizer: object, model: object, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_input_tokens = 900

    def _summarize_chunk(self, text: str, config: GenerationConfig) -> str:
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            try:
                import torch
                inference_context = torch.inference_mode()
            except ImportError:
                from contextlib import nullcontext

                inference_context = nullcontext()
            with inference_context:
                output = self.model.generate(
                    **inputs,
                    min_length=config.min_length,
                    max_length=config.max_length,
                    num_beams=4,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        except Exception as exc:
            raise SummarizationError("The model failed while generating a summary.") from exc

    def summarize(self, text: str, length: SummaryLength = SummaryLength.MEDIUM) -> str:
        if not text.strip():
            raise SummarizationError("There is no text to summarize.")
        config = _CONFIGS[SummaryLength(length)]
        chunks = chunk_text(text, self.tokenizer, self.max_input_tokens)
        partials = [self._summarize_chunk(chunk, config) for chunk in chunks]
        # Summarize the complete set of partial summaries recursively, never silently
        # dropping chunks. Shorter intermediate generations keep the final input bounded.
        while len(partials) > 1:
            combined = " ".join(partials)
            if len(chunks := chunk_text(combined, self.tokenizer, self.max_input_tokens)) == 1:
                partials = [self._summarize_chunk(chunks[0], config)]
            else:
                partials = [self._summarize_chunk(chunk, config) for chunk in chunks]
        return partials[0] if partials else ""
