"""Text embeddings for the local memory, sized for a CLI rather than a service.

The embedder is model2vec's static vocabulary lookup, not a transformer. That
trade is deliberate: inference is numpy-only (no torch, no GPU, no second
runtime), the default model is ~30MB fetched once, and encoding a batch of
candidates costs milliseconds — which is the right weight for a tool that runs
on a laptop between two network calls. What is given up is contextual quality:
a static model cannot tell "java the island" from "java the language". For the
jobs memory does — near-duplicate detection and recall over one user's own
evidence — that ceiling does not bind; two cross-posts of the same complaint
are near-identical strings, and no context window is needed to see it.
"""

import math
from typing import Any, Protocol

from app.config import get_settings
from app.utils.errors import MemoryUnavailableError


class Embedder(Protocol):
    """Anything that can turn texts into fixed-width vectors.

    A protocol rather than a base class so tests can inject a deterministic
    fake without importing (or downloading) any real model.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed each text, preserving order. One vector per input."""
        ...


class StaticModelEmbedder:
    """A model2vec static model, loaded lazily on the first encode.

    Lazy because construction happens on every run that has memory enabled,
    including runs that turn out to have nothing to embed — the import and the
    Hugging Face cache lookup are paid only when a vector is actually needed.

    Raises:
        MemoryUnavailableError: If the model cannot be loaded (model2vec not
            installed, offline with a cold cache, a bad model name) or cannot
            encode. Callers degrade to running without memory; they never crash
            a run over it.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed each text with the static model, loading it on first use.

        Encoding is wrapped as carefully as loading, and for the same reason.
        model2vec tokenises in parallel, so a failure here surfaces as whatever
        the multiprocessing layer felt like raising — `ValueError: bad value(s)
        in fds_to_keep` is one seen in the wild — and none of those types mean
        anything to a caller beyond "no vectors". Left unconverted, one of them
        took down a four-minute `collect-evidence` attempt that had already
        fetched everything it needed.
        """
        model = self._load()
        try:
            vectors: list[list[float]] = model.encode(texts).tolist()
        except Exception as exc:
            raise MemoryUnavailableError(
                f"embedding model {self._model_name!r} could not encode {len(texts)} text(s): {exc}"
            ) from exc
        return vectors

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from model2vec import StaticModel
        except ImportError as exc:
            raise MemoryUnavailableError(
                "model2vec is not installed; semantic memory is off"
            ) from exc
        try:
            self._model = StaticModel.from_pretrained(self._model_name)
        except Exception as exc:
            # From here down is hf-hub territory: offline, a cold cache, a typo'd
            # model name. All of them mean the same thing to a caller — no vectors.
            raise MemoryUnavailableError(
                f"could not load embedding model {self._model_name!r}: {exc}"
            ) from exc
        return self._model


def default_embedder() -> Embedder:
    """The embedder the app uses when nothing is injected: the configured model."""
    return StaticModelEmbedder(get_settings().embedding_model)


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. A zero vector is similar to nothing."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0.0:
        return 0.0
    return dot / norm


__all__ = ["Embedder", "StaticModelEmbedder", "cosine", "default_embedder"]
