"""Score every retrieval configuration against the relevance judgments.

Reads data/labels/label_sheet.csv (annotated per ANNOTATION_GUIDELINES.md)
and computes P@10 and nDCG@10 for each configuration cell recorded in
eval_results.json and calibration_results.json. Unlabeled retrieved items are
treated as not relevant and counted in a warning. Output:
data/relevance_scores.json plus a printed summary table.
"""

import csv
import json
import math
from collections import defaultdict

from config import DATA_DIR, TOP_K

LABELS_PATH = DATA_DIR / "labels" / "label_sheet.csv"
RESULTS_PATH = DATA_DIR / "relevance_scores.json"


def load_qrels() -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    unlabeled = 0
    with open(LABELS_PATH) as f:
        for row in csv.DictReader(f):
            raw = row["label"].strip()
            if raw not in ("0", "1"):
                unlabeled += 1
                continue
            qrels[row["query_id"]][row["item_key"]] = int(raw)
    if unlabeled:
        print(f"WARNING: {unlabeled} rows without a 0/1 label; treated as not relevant")
    return qrels


def ndcg_at_k(gains: list[int], k: int) -> float:
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def score_cell(cell: dict, qrels: dict) -> dict:
    p_at_k, ndcgs = [], []
    for pq in cell["per_query"]:
        judgments = qrels.get(pq["query_id"], {})
        gains = [judgments.get(key, 0) for key in pq["items"]]
        p_at_k.append(sum(gains[:TOP_K]) / TOP_K)
        # ideal ranking considers all judged-relevant items for the query
        all_rel = sorted(judgments.values(), reverse=True)
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:TOP_K]))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(all_rel[:TOP_K]))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    n = len(p_at_k)
    return {
        "p_at_10": round(sum(p_at_k) / n, 3),
        "ndcg_at_10": round(sum(ndcgs) / n, 3),
    }


def iter_cells():
    with open(DATA_DIR / "eval_results.json") as f:
        ev = json.load(f)
    for enc, data in ev["encoders"].items():
        yield f"{enc}/raw/en", data["en_queries"]
        yield f"{enc}/raw/id", data["id_queries"]
    yield "bm25/-/en", ev["bm25"]["en_queries"]
    yield "bm25/-/id", ev["bm25"]["id_queries"]

    with open(DATA_DIR / "calibration_results.json") as f:
        cal = json.load(f)
    for enc, modes in cal["encoders"].items():
        for mode in ("raw", "calibrated"):
            if mode == "raw":
                continue  # raw cells already covered by eval_results
            yield f"{enc}/calibrated/en", modes[mode]["en_queries"]
            yield f"{enc}/calibrated/id", modes[mode]["id_queries"]


def main():
    qrels = load_qrels()
    judged = sum(len(v) for v in qrels.values())
    relevant = sum(sum(v.values()) for v in qrels.values())
    print(f"qrels: {judged} judged items, {relevant} relevant ({relevant/judged*100:.0f}%)\n"
          if judged else "No labels found - fill in label_sheet.csv first.\n")
    if not judged:
        return

    results = {}
    print(f"{'configuration':<24} {'P@10':>7} {'nDCG@10':>9}")
    print("-" * 42)
    for name, cell in iter_cells():
        scores = score_cell(cell, qrels)
        results[name] = scores
        print(f"{name:<24} {scores['p_at_10']:>7} {scores['ndcg_at_10']:>9}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
