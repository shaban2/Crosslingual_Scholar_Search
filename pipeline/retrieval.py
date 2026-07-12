import pickle
from pathlib import Path

import faiss
import numpy as np

from config import DEFAULT_ENCODER, TOP_K, index_path as encoder_index_path
from pipeline.encoders import get_text_model


_index_cache = {}


def build_faiss_index(
    all_embeddings: list[list[float]],
    all_metadata: list[dict],
    index_path: str | Path,
):
    dim = len(all_embeddings[0])
    index = faiss.IndexFlatIP(dim)
    embeddings_np = np.array(all_embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    data = {
        "index": index,
        "metadata": all_metadata,
    }
    with open(index_path, "wb") as f:
        pickle.dump(data, f)
    _index_cache.pop(str(index_path), None)
    print(f"FAISS index saved to {index_path} ({index.ntotal} vectors)")


def load_index(encoder: str = DEFAULT_ENCODER, index_path: str | Path | None = None):
    key = str(index_path or encoder_index_path(encoder))
    if key not in _index_cache:
        with open(key, "rb") as f:
            data = pickle.load(f)
        _index_cache[key] = (data["index"], data["metadata"])
    return _index_cache[key]


def embed_query(query: str, encoder: str = DEFAULT_ENCODER) -> list[float]:
    model = get_text_model(encoder)
    return model.encode([query])[0].tolist()


def search(query: str, k: int = TOP_K, modality: str | None = None, encoder: str = DEFAULT_ENCODER):
    index, metadata = load_index(encoder)
    query_vec = np.array([embed_query(query, encoder)]).astype("float32")
    faiss.normalize_L2(query_vec)

    if modality:
        scores, indices = index.search(query_vec, index.ntotal)
        results = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            m = metadata[idx]
            if m.get("modality") != modality:
                continue
            dedup_key = m.get("image_path", str(idx))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            results.append({
                "score": float(score),
                "metadata": m,
            })
            if len(results) >= k:
                break
        return results

    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "metadata": metadata[idx],
        })
    return results
