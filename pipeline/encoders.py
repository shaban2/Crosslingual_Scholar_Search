"""Shared encoder registry.

Each encoder config pairs a text model with an image model whose embedding
spaces are aligned. All pipeline modules load models through this registry so
each model is loaded exactly once per process (loading duplicate copies
alongside FAISS can segfault on macOS due to duplicate OpenMP runtimes).
"""

from sentence_transformers import SentenceTransformer

from config import ENCODERS

_models: dict[str, SentenceTransformer] = {}


def _load(model_name: str) -> SentenceTransformer:
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def get_text_model(encoder: str) -> SentenceTransformer:
    return _load(ENCODERS[encoder]["text"])


def get_image_model(encoder: str) -> SentenceTransformer:
    return _load(ENCODERS[encoder]["image"])
