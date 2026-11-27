"""Re-rank retrieved chunks with a cross-encoder.

Vector search compares two vectors that were made separately: one for the question, one
for the chunk. That is fast, but rough. A cross-encoder reads the question and the chunk
*together* and scores how well the chunk answers the question. It is much more accurate,
but too slow to run on every chunk, so we only run it on the ~20 candidates that vector
search found, and keep the best few.

The model is loaded lazily (on first use), like the embedding model.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Callable, Protocol

from app.domain import RetrievedChunk

logger = logging.getLogger(__name__)


class PairScorer(Protocol):
    """The subset of CrossEncoder we use; lets tests inject a fake model."""

    def predict(self, inputs: Any, **kwargs: Any) -> Any: ...


class RerankService:
    def __init__(self, model_name: str, batch_size: int = 32, model: PairScorer | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model
        self._activation: Callable | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> PairScorer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import CrossEncoder  # heavy import

                    logger.info("Loading re-ranking model '%s'...", self.model_name)
                    # Sigmoid turns the raw score into 0..1, which is easier to read.
                    self._activation = torch.sigmoid
                    self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Score every chunk against the query and return the best `top_k`, best first."""
        if not chunks:
            return []
        model = self._get_model()
        scores = model.predict(
            [(query, c.text) for c in chunks],
            batch_size=self.batch_size,
            activation_fn=self._activation,
            show_progress_bar=False,
        )
        scored = [dataclasses.replace(c, rerank_score=float(s)) for c, s in zip(chunks, scores)]
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        logger.info(
            "Re-ranked %d candidate(s); kept %d (best=%.3f)", len(chunks), min(top_k, len(scored)), scored[0].rerank_score
        )
        return scored[:top_k]

    def warm_up(self) -> None:
        self._get_model()
