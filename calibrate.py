"""Per-group score calibration experiment.

Raw cosine similarities are not comparable across (modality, language)
groups: each group's score distribution has its own mean and spread (see
analyze_gap.py), so a shared top-K is dominated by whichever group scores
structurally higher. This experiment z-scores each query's similarities
WITHIN each (modality, language) group before ranking, and asks whether a
single calibrated index produces the balanced, language-fair mixed list that
raw retrieval fails to produce.

For each encoder x query language, retrieval is run raw and calibrated, and
the same metrics as evaluate.py are reported, plus EN<->ID consistency.
Results go to data/calibration_results.json.
"""

import json
from collections import Counter, defaultdict

import faiss
import numpy as np

from config import DATA_DIR, ENCODERS, TOP_K
from evaluate import en_id_consistency, item_key, load_queries
from pipeline.retrieval import embed_query, load_index

RESULTS_PATH = DATA_DIR / "calibration_results.json"


def group_of(meta: dict) -> tuple[str, str]:
    return (meta.get("modality"), meta.get("language"))


def rank_rows(sims: np.ndarray, metadata: list[dict], calibrated: bool) -> list[list[int]]:
    """Return top-K item indices per query row, raw or z-scored per group."""
    if calibrated:
        groups = defaultdict(list)
        for i, m in enumerate(metadata):
            groups[group_of(m)].append(i)
        scores = np.empty_like(sims)
        for idx in groups.values():
            cols = np.array(idx)
            vals = sims[:, cols]
            mean = vals.mean(axis=1, keepdims=True)
            std = vals.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            scores[:, cols] = (vals - mean) / std
    else:
        scores = sims
    return [list(np.argsort(-row)[:TOP_K]) for row in scores]


def cell_stats(queries: list[dict], lang_field: str, top_indices: list[list[int]],
               metadata: list[dict]) -> dict:
    per_query = []
    modality_counts = Counter()
    language_counts = Counter()
    coverage = Counter()
    for q, idxs in zip(queries, top_indices):
        metas = [metadata[i] for i in idxs]
        mods = Counter(m.get("modality") for m in metas)
        langs = Counter(m.get("language") for m in metas)
        modality_counts.update(mods)
        language_counts.update(langs)
        if mods.get("figure", 0) > 0:
            coverage["figure"] += 1
        if mods.get("transcript", 0) > 0:
            coverage["transcript"] += 1
        if langs.get("en", 0) > 0:
            coverage["en_content"] += 1
        if langs.get("id", 0) > 0:
            coverage["id_content"] += 1
        per_query.append({
            "query_id": q["id"],
            "query": q[lang_field],
            "modalities": dict(mods),
            "languages": dict(langs),
            "items": [item_key(m) for m in metas],
        })
    n = len(queries)
    return {
        "per_query": per_query,
        "modality_counts": {m: modality_counts.get(m, 0) for m in ("text", "figure", "transcript")},
        "language_counts": {l: language_counts.get(l, 0) for l in ("en", "id")},
        "coverage": {
            "queries_with_figure": coverage["figure"],
            "queries_with_transcript": coverage["transcript"],
            "queries_with_en_content": coverage["en_content"],
            "queries_with_id_content": coverage["id_content"],
            "n_queries": n,
        },
    }


def print_cell(name: str, stats: dict):
    n = stats["coverage"]["n_queries"]
    m, l, c = stats["modality_counts"], stats["language_counts"], stats["coverage"]
    print(f"  {name:<24} text {m['text']:>3} | fig {m['figure']:>2} | trans {m['transcript']:>3}"
          f" || en {l['en']:>3} | id {l['id']:>3}"
          f" || cov: fig {c['queries_with_figure']}/{n}"
          f" en {c['queries_with_en_content']}/{n} id {c['queries_with_id_content']}/{n}")


def calibrate():
    queries = load_queries()
    results = {"top_k": TOP_K, "encoders": {}}

    for encoder in sorted(ENCODERS):
        index, metadata = load_index(encoder)
        vectors = index.reconstruct_n(0, index.ntotal)
        print(f"\n== {encoder} ==")
        enc_results = {}
        cells = {}
        for lang_field, label in (("en", "en_queries"), ("id_", "id_queries")):
            qm = np.array([embed_query(q[lang_field], encoder) for q in queries]).astype("float32")
            faiss.normalize_L2(qm)
            sims = qm @ vectors.T
            for calibrated, mode in ((False, "raw"), (True, "calibrated")):
                tops = rank_rows(sims, metadata, calibrated)
                stats = cell_stats(queries, lang_field, tops, metadata)
                cells[(label, mode)] = stats
                print_cell(f"{label}/{mode}", stats)
        for mode in ("raw", "calibrated"):
            cons = en_id_consistency(cells[("en_queries", mode)], cells[("id_queries", mode)])
            enc_results[mode] = {
                "en_queries": cells[("en_queries", mode)],
                "id_queries": cells[("id_queries", mode)],
                "en_id_consistency": cons,
            }
            print(f"  {mode}: EN<->ID Jaccard {cons['mean_jaccard']} | overlap@{TOP_K} {cons['mean_overlap_at_k']}")
        results["encoders"][encoder] = enc_results

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    calibrate()
