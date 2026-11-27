"""One "embedding space" per embedding model: the model itself plus its own Chroma collection.

Vectors made by different models have different sizes (384 vs 768 numbers) and different
meanings, so they must never share a collection. Each document is indexed with the model
the user picked at upload time; at question time the question is embedded once per model
and searched in that model's collection.

The default model keeps the original collection name, so existing data keeps working.
Other models get "<collection>-<short hash of model name>" (Chroma allows only a few
characters in names, and model names contain "/").
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.exceptions import InvalidModelError
from app.repositories.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService


@dataclass(frozen=True)
class EmbeddingSpace:
    model_name: str
    embedder: EmbeddingService
    store: VectorStore


class EmbeddingSpaces:
    def __init__(
        self,
        default_embedder: EmbeddingService,
        default_store: VectorStore,
        persist_dir: Path,
        base_collection: str,
        allowed_models: list[str],
        embedder_factory: Callable[[str], EmbeddingService],
    ) -> None:
        self.default_model = default_embedder.model_name
        self.allowed_models = list(dict.fromkeys([self.default_model, *allowed_models]))
        self._persist_dir = persist_dir
        self._base_collection = base_collection
        self._embedder_factory = embedder_factory
        self._spaces = {self.default_model: EmbeddingSpace(self.default_model, default_embedder, default_store)}
        self._lock = threading.Lock()

    def resolve(self, model_name: str | None) -> str:
        """Return the model to use (default when empty); refuse models not in the allowed list."""
        name = (model_name or "").strip() or self.default_model
        if name not in self.allowed_models:
            raise InvalidModelError(
                f"Embedding model '{name}' is not allowed. Choose one of: {', '.join(self.allowed_models)}."
            )
        return name

    def get(self, model_name: str | None) -> EmbeddingSpace:
        """The space for a model, created on first use (the model itself loads lazily)."""
        name = self.resolve(model_name)
        with self._lock:
            if name not in self._spaces:
                store = VectorStore(self._persist_dir, self._collection_name(name), name)
                self._spaces[name] = EmbeddingSpace(name, self._embedder_factory(name), store)
            return self._spaces[name]

    def count(self) -> int:
        """Chunks stored across every allowed model."""
        return sum(self.get(name).store.count() for name in self.allowed_models)

    def _collection_name(self, model_name: str) -> str:
        digest = hashlib.sha256(model_name.encode()).hexdigest()[:12]
        return f"{self._base_collection}-{digest}"
