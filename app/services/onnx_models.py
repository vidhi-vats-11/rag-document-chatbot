"""Load Sentence Transformers models with fastembed (ONNX Runtime) instead of PyTorch.

Why: PyTorch is ~700 MB, too big for small hosting plans (Vercel allows 500 MB in total).
ONNX Runtime runs the *same* models in ~50 MB. We checked that the vectors and scores
match the PyTorch versions (difference < 0.0003), so already-indexed documents stay valid.

Most Sentence Transformers models publish an ONNX copy (`onnx/model.onnx`) on Hugging Face.
fastembed knows some of them already; any other one is registered here the first time it
is used, reading its pooling and size from the model's own config files.

The two small classes below give the services the same `encode` / `predict` methods they
had with sentence-transformers, so tests can still swap in fakes.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_ONNX_FILE = "onnx/model.onnx"
_register_lock = threading.Lock()


def _cache_dir() -> str:
    # fastembed's default cache is a temp folder that the OS may clear; keep models instead.
    return os.environ.get("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "fastembed")


def _hf_json(repo: str, filename: str) -> dict:
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(repo, filename)).read_text())


def _ensure_embedding_model(name: str) -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    with _register_lock:
        if name in {m["model"] for m in TextEmbedding.list_supported_models()}:
            return
        dim = _hf_json(name, "config.json")["hidden_size"]
        try:
            uses_cls = _hf_json(name, "1_Pooling/config.json").get("pooling_mode_cls_token", False)
        except Exception:  # no pooling file: Sentence Transformers' default is mean pooling
            uses_cls = False
        TextEmbedding.add_custom_model(
            model=name,
            pooling=PoolingType.CLS if uses_cls else PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=name),
            dim=dim,
            model_file=_ONNX_FILE,
        )


def _ensure_cross_encoder(name: str) -> None:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from fastembed.common.model_description import ModelSource

    with _register_lock:
        if name in {m["model"] for m in TextCrossEncoder.list_supported_models()}:
            return
        TextCrossEncoder.add_custom_model(model=name, sources=ModelSource(hf=name), model_file=_ONNX_FILE)


class OnnxSentenceEncoder:
    """Embedding model: `encode(texts)` returns one L2-normalised vector per text."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        _ensure_embedding_model(model_name)
        self._model = TextEmbedding(model_name, cache_dir=_cache_dir())

    def encode(self, inputs: Iterable[str], batch_size: int = 32, **_: Any) -> np.ndarray:
        vectors = np.array(list(self._model.embed(list(inputs), batch_size=batch_size)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


class OnnxCrossEncoder:
    """Re-ranking model: `predict(pairs)` returns a 0..1 score per (question, text) pair."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _ensure_cross_encoder(model_name)
        self._model = TextCrossEncoder(model_name, cache_dir=_cache_dir())

    def predict(self, inputs: Iterable[tuple[str, str]], batch_size: int = 32, **_: Any) -> np.ndarray:
        raw = np.array(list(self._model.rerank_pairs(list(inputs), batch_size=batch_size)), dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-raw))  # sigmoid: raw score → 0..1, easier to read
