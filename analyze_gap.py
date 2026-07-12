"""Similarity-distribution analysis: the language gap and the modality gap.

For each encoder and each query language, computes cosine similarity between
every query and every indexed item, grouped by (modality, content language).
The difference between same-language and cross-language group means is the
language gap; the text-vs-figure difference is the modality gap. Results go
to data/gap_analysis.json.
"""

import json
from collections import defaultdict

import faiss
import numpy as np

from config import DATA_DIR, ENCODERS
from evaluate import load_queries
from pipeline.retrieval import embed_query, load_index

RESULTS_PATH = DATA_DIR / "gap_analysis.json"

GROUPS = [
    ("text", "en"),
    ("figure", "en"),
    ("transcript", "en"),
    ("transcript", "id"),
]


def query_matrix(queries: list[dict], lang_field: str, encoder: str) -> np.ndarray:
    vecs = np.array([embed_query(q[lang_field], encoder) for q in queries]).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def group_stats(sims: np.ndarray, metadata: list[dict]) -> dict:
    cols = defaultdict(list)
    for i, m in enumerate(metadata):
        cols[(m.get("modality"), m.get("language"))].append(i)
    stats = {}
    for group in GROUPS:
        idx = cols.get(group)
        if not idx:
            continue
        vals = sims[:, idx]
        stats[f"{group[0]}_{group[1]}"] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "max": round(float(vals.max()), 4),
            "n_items": len(idx),
        }
    return stats


def analyze():
    queries = load_queries()
    results = {}
    for encoder in sorted(ENCODERS):
        index, metadata = load_index(encoder)
        vectors = index.reconstruct_n(0, index.ntotal)
        per_lang = {}
        for lang_field, label in (("en", "en_queries"), ("id_", "id_queries")):
            qm = query_matrix(queries, lang_field, encoder)
            sims = qm @ vectors.T
            per_lang[label] = group_stats(sims, metadata)
        results[encoder] = per_lang

        print(f"\n== {encoder} ==")
        header = f"{'group':<16}" + "".join(f"{label:>22}" for label in per_lang)
        print(header + "   (mean +/- std)")
        for group in GROUPS:
            key = f"{group[0]}_{group[1]}"
            row = f"{key:<16}"
            for label in per_lang:
                s = per_lang[label].get(key)
                row += f"{s['mean']:>12} +/- {s['std']:<6}" if s else f"{'-':>22}"
            print(row)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    analyze()
