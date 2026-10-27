"""Turn text into vectors with a Sentence Transformers model.

The model is loaded lazily (on first use) because loading takes a few
seconds and downloads weights the very first time. Vectors are L2-normalised,
so cosine similarity is a simple dot product and scores are comparable.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SentenceEncoder(Protocol):
    """The subset of SentenceTransformer we use; lets tests inject a fake model."""

    def encode(self, inputs: Any, **kwargs: Any) -> Any: ...


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        model: SentenceEncoder | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model
        self._lock = threading.Lock()

    def _get_model(self) -> SentenceEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer  # heavy import

                    logger.info("Loading embedding model '%s'...", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def warm_up(self) -> None:
        """Load the model now (used at startup so the first request is not slow)."""
        self._get_model()
