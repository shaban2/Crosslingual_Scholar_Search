"""Run the versioned core-revision experiments without overwriting legacy results.

The script evaluates original and formal Indonesian query sets at k=5,10,20,
stores per-query rankings, retrieval summaries, similarity-group statistics,
provisional LLM-label effectiveness, and a paired robustness comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config import DATA_DIR, ENCODERS
from evaluate import item_key, item_text
from pipeline.retrieval import embed_query, load_index

CUTOFFS = (5, 10, 20)
SEED = 20260718
RESULTS_DIR = DATA_DIR / "revision_results"
QUERY_SETS = {
    "original": DATA_DIR / "queries.json",
    "formal-id-v1": DATA_DIR / "queries_formal_v1.json",
}
GROUPS = (("text", "en"), ("figure", "en"), ("transcript", "en"), ("transcript", "id"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_query_set(name: str) -> list[dict]:
    with QUERY_SETS[name].open() as handle:
        return json.load(handle)["queries"]


def validate_query_sets() -> None:
    original = load_query_set("original")
    formal = load_query_set("formal-id-v1")
    assert len(original) == len(formal) == 15
    assert [q["id"] for q in original] == [q["id"] for q in formal]
    assert [q["topic"] for q in original] == [q["topic"] for q in formal]
    assert [q["en"] for q in original] == [q["en"] for q in formal]
    with (DATA_DIR / "query_revision_audit.csv").open() as handle:
        audit = list(csv.DictReader(handle))
    assert [row["query_id"] for row in audit] == [q["id"] for q in original]
    assert all(row["semantic_change"] == "none" for row in audit)


def metadata_block(query_set: str) -> dict:
    return {
        "schema_version": 1,
        "query_set_id": query_set,
        "query_file": str(QUERY_SETS[query_set].relative_to(DATA_DIR.parent)),
        "query_sha256": sha256(QUERY_SETS[query_set]),
        "corpus_index_sha256": {
            encoder: sha256(DATA_DIR / f"faiss_index_{encoder}.pkl") for encoder in sorted(ENCODERS)
        },
        "cutoffs": list(CUTOFFS),
        "random_seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "models": ENCODERS,
        "relevance_assessor": "single LLM; provisional, not human-validated",
    }


def summarize_rankings(rankings: list[dict], k: int) -> dict:
    modality_counts, language_counts, coverage = Counter(), Counter(), Counter()
    per_query = []
    for row in rankings:
        items = row["results"][:k]
        mods = Counter(item["modality"] for item in items)
        langs = Counter(item["language"] for item in items)
        modality_counts.update(mods)
        language_counts.update(langs)
        for modality in ("figure", "transcript"):
            if mods[modality]:
                coverage[modality] += 1
        for language in ("en", "id"):
            if langs[language]:
                coverage[f"{language}_content"] += 1
        per_query.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "modalities": dict(mods),
            "languages": dict(langs),
            "items": [item["item_key"] for item in items],
            "scores": [item["score"] for item in items],
        })
    return {
        "total_items": len(rankings) * k,
        "modality_counts": {m: modality_counts[m] for m in ("text", "figure", "transcript")},
        "language_counts": {l: language_counts[l] for l in ("en", "id")},
        "coverage": {
            "queries_with_figure": coverage["figure"],
            "queries_with_transcript": coverage["transcript"],
            "queries_with_en_content": coverage["en_content"],
            "queries_with_id_content": coverage["id_content"],
            "n_queries": len(rankings),
        },
        "per_query": per_query,
    }


def consistency(en_cell: dict, id_cell: dict) -> dict:
    id_rows = {row["query_id"]: row for row in id_cell["per_query"]}
    per_query, jaccards, overlaps = {}, [], []
    for en_row in en_cell["per_query"]:
        a, b = set(en_row["items"]), set(id_rows[en_row["query_id"]]["items"])
        intersection = len(a & b)
        jaccard = intersection / len(a | b) if a or b else 0.0
        overlap = intersection / len(a) if a else 0.0
        per_query[en_row["query_id"]] = {"jaccard": jaccard, "overlap_at_k": overlap}
        jaccards.append(jaccard)
        overlaps.append(overlap)
    return {
        "mean_jaccard": float(np.mean(jaccards)),
        "mean_overlap_at_k": float(np.mean(overlaps)),
        "per_query": per_query,
    }


def rankings_from_scores(queries: list[dict], lang: str, scores: np.ndarray, metadata: list[dict]) -> list[dict]:
    order = np.argsort(-scores, axis=1)[:, : max(CUTOFFS)]
    rows = []
    for query, indices, score_row in zip(queries, order, scores):
        results = []
        for idx in indices:
            meta = metadata[int(idx)]
            results.append({
                "item_key": item_key(meta),
                "score": float(score_row[int(idx)]),
                "modality": meta.get("modality"),
                "language": meta.get("language"),
            })
        rows.append({"query_id": query["id"], "query": query[lang], "results": results})
    return rows


def calibrate_scores(scores: np.ndarray, metadata: list[dict]) -> np.ndarray:
    calibrated = np.empty_like(scores)
    groups = defaultdict(list)
    for idx, meta in enumerate(metadata):
        groups[(meta.get("modality"), meta.get("language"))].append(idx)
    for indices in groups.values():
        cols = np.asarray(indices)
        values = scores[:, cols]
        std = values.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        calibrated[:, cols] = (values - values.mean(axis=1, keepdims=True)) / std
    return calibrated


def group_similarity(scores: np.ndarray, metadata: list[dict]) -> dict:
    grouped = defaultdict(list)
    for idx, meta in enumerate(metadata):
        grouped[(meta.get("modality"), meta.get("language"))].append(idx)
    result = {}
    for modality, language in GROUPS:
        values = scores[:, grouped[(modality, language)]]
        result[f"{modality}_{language}"] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "max": float(values.max()), "n_items": values.shape[1],
        }
    return result


class BM25:
    def __init__(self, metadata: list[dict]):
        self.items = [meta for meta in metadata if item_text(meta).strip()]
        self.model = BM25Okapi([item_text(meta).lower().split() for meta in self.items])

    def rankings(self, queries: list[dict], lang: str) -> list[dict]:
        rows = []
        for query in queries:
            scores = self.model.get_scores(query[lang].lower().split())
            indices = np.argsort(-scores)[: max(CUTOFFS)]
            results = []
            for idx in indices:
                meta = self.items[int(idx)]
                results.append({"item_key": item_key(meta), "score": float(scores[int(idx)]),
                                "modality": meta.get("modality"), "language": meta.get("language")})
            rows.append({"query_id": query["id"], "query": query[lang], "results": results})
        return rows


def cells_for_rankings(en_rankings: list[dict], id_rankings: list[dict]) -> dict:
    output = {}
    for k in CUTOFFS:
        en_cell, id_cell = summarize_rankings(en_rankings, k), summarize_rankings(id_rankings, k)
        output[str(k)] = {"en_queries": en_cell, "id_queries": id_cell,
                          "en_id_consistency": consistency(en_cell, id_cell)}
    return output


def load_qrels() -> dict[str, dict[str, int]]:
    qrels = defaultdict(dict)
    with (DATA_DIR / "labels" / "label_sheet.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["label"].strip() in {"0", "1"}:
                qrels[row["query_id"]][row["item_key"]] = int(row["label"])
    return qrels


def effectiveness(cell: dict, qrels: dict, k: int) -> dict:
    per_query = {}
    for row in cell["per_query"]:
        judgments = qrels[row["query_id"]]
        labels = [judgments.get(key) for key in row["items"][:k]]
        gains = [label or 0 for label in labels]
        ideal = sorted(judgments.values(), reverse=True)[:k]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        per_query[row["query_id"]] = {
            "precision": sum(gains) / k,
            "ndcg": dcg / idcg if idcg else 0.0,
            "judgment_coverage": sum(label is not None for label in labels) / k,
        }
    return {
        "precision": float(np.mean([v["precision"] for v in per_query.values()])),
        "ndcg": float(np.mean([v["ndcg"] for v in per_query.values()])),
        "judgment_coverage": float(np.mean([v["judgment_coverage"] for v in per_query.values()])),
        "per_query": per_query,
        "status": "provisional_single_llm_assessor",
    }


def add_effectiveness(result: dict) -> None:
    qrels = load_qrels()
    for config in result["configurations"].values():
        for k, cells in config["cutoffs"].items():
            cells["effectiveness"] = {
                "en": effectiveness(cells["en_queries"], qrels, int(k)),
                "id": effectiveness(cells["id_queries"], qrels, int(k)),
            }


def run_query_set(name: str) -> dict:
    queries = load_query_set(name)
    result = {"metadata": metadata_block(name), "configurations": {}, "similarity_groups": {}}
    for encoder in sorted(ENCODERS):
        index, metadata = load_index(encoder)
        vectors = index.reconstruct_n(0, index.ntotal)
        matrices = {}
        for lang in ("en", "id_"):
            matrix = np.asarray([embed_query(query[lang], encoder) for query in queries], dtype="float32")
            faiss.normalize_L2(matrix)
            matrices[lang] = matrix @ vectors.T
        result["similarity_groups"][encoder] = {
            "en_queries": group_similarity(matrices["en"], metadata),
            "id_queries": group_similarity(matrices["id_"], metadata),
        }
        for mode, transform in (("raw", lambda x: x), ("calibrated", lambda x: calibrate_scores(x, metadata))):
            en_rankings = rankings_from_scores(queries, "en", transform(matrices["en"]), metadata)
            id_rankings = rankings_from_scores(queries, "id_", transform(matrices["id_"]), metadata)
            result["configurations"][f"{encoder}/{mode}"] = {"cutoffs": cells_for_rankings(en_rankings, id_rankings)}
    _, metadata = load_index("clip")
    bm25 = BM25(metadata)
    result["configurations"]["bm25/raw"] = {
        "cutoffs": cells_for_rankings(bm25.rankings(queries, "en"), bm25.rankings(queries, "id_"))
    }
    add_effectiveness(result)
    return result


def bootstrap_difference(original: list[float], formal: list[float], rng: random.Random) -> dict:
    differences = [b - a for a, b in zip(original, formal)]
    n = len(differences)
    samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000)]
    mean = float(np.mean(differences))
    std = float(np.std(differences, ddof=1))
    return {
        "mean_difference_formal_minus_original": mean,
        "ci_95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "cohen_dz": mean / std if std else 0.0,
        "per_query_differences": differences,
    }


def compare(original: dict, formal: dict) -> dict:
    rng = random.Random(SEED)
    output = {"metadata": {"random_seed": SEED, "bootstrap_samples": 10000,
                            "comparison": "formal-id-v1 minus original"}, "configurations": {}}
    for name in original["configurations"]:
        output["configurations"][name] = {}
        for k in map(str, CUTOFFS):
            a, b = original["configurations"][name]["cutoffs"][k], formal["configurations"][name]["cutoffs"][k]
            entry = {}
            for lang in ("en", "id"):
                metric = "ndcg"
                av = [v[metric] for v in a["effectiveness"][lang]["per_query"].values()]
                bv = [v[metric] for v in b["effectiveness"][lang]["per_query"].values()]
                entry[f"{lang}_ndcg"] = bootstrap_difference(av, bv, rng)
            aj = [v["jaccard"] for v in a["en_id_consistency"]["per_query"].values()]
            bj = [v["jaccard"] for v in b["en_id_consistency"]["per_query"].values()]
            entry["paired_query_jaccard"] = bootstrap_difference(aj, bj, rng)
            output["configurations"][name][k] = entry
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-set", choices=[*QUERY_SETS, "all"], default="all")
    args = parser.parse_args()
    validate_query_sets()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = list(QUERY_SETS) if args.query_set == "all" else [args.query_set]
    results = {}
    for name in names:
        results[name] = run_query_set(name)
        with (RESULTS_DIR / f"{name}.json").open("w") as handle:
            json.dump(results[name], handle, ensure_ascii=False, indent=2)
        print(f"Wrote {RESULTS_DIR / f'{name}.json'}")
    if args.query_set == "all":
        comparison = compare(results["original"], results["formal-id-v1"])
        with (RESULTS_DIR / "comparison.json").open("w") as handle:
            json.dump(comparison, handle, ensure_ascii=False, indent=2)
        print(f"Wrote {RESULTS_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
