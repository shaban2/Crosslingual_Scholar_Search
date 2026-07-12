"""Generate relevance-judgment sheets.

Pools the union of items retrieved by every configuration (encoder x query
language x raw/calibrated, plus BM25) for each query, and writes one CSV for
the annotator: mark `label` 1 (relevant to the query) or 0 (not relevant).
Judge relevance against the MEANING of the query pair (EN and ID are the same
question); an item in either language can be relevant to either query version.
Output: data/labels/label_sheet.csv
"""

import csv
import json

from config import DATA_DIR
from evaluate import item_text, load_queries

LABELS_DIR = DATA_DIR / "labels"
SOURCES = [
    ("eval_results.json", lambda d: [
        cell for enc in d["encoders"].values() for cell in (enc["en_queries"], enc["id_queries"])
    ] + [d["bm25"]["en_queries"], d["bm25"]["id_queries"]]),
    ("calibration_results.json", lambda d: [
        enc[mode][cell]
        for enc in d["encoders"].values()
        for mode in ("raw", "calibrated")
        for cell in ("en_queries", "id_queries")
    ]),
]


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    queries = {q["id"]: q for q in load_queries()}

    # union of item keys per query across all configurations
    pool: dict[str, set[str]] = {qid: set() for qid in queries}
    for filename, extract in SOURCES:
        with open(DATA_DIR / filename) as f:
            data = json.load(f)
        for cell in extract(data):
            for pq in cell["per_query"]:
                pool[pq["query_id"]].update(pq["items"])

    # snippet lookup from the index metadata
    from pipeline.retrieval import load_index
    from evaluate import item_key
    _, metadata = load_index("clip")
    meta_by_key = {item_key(m): m for m in metadata}

    out = LABELS_DIR / "label_sheet.csv"
    n_rows = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query_en", "query_id_indonesian",
                    "item_key", "modality", "language", "snippet", "label"])
        for qid in sorted(pool):
            q = queries[qid]
            for key in sorted(pool[qid]):
                m = meta_by_key.get(key, {})
                snippet = item_text(m)[:250].replace("\n", " ")
                w.writerow([qid, q["en"], q["id_"], key,
                            m.get("modality", "?"), m.get("language", "?"), snippet, ""])
                n_rows += 1
    print(f"{out}: {n_rows} items to label across {len(pool)} queries")
    per_q = [len(v) for v in pool.values()]
    print(f"per query: min {min(per_q)}, mean {sum(per_q)/len(per_q):.0f}, max {max(per_q)}")


if __name__ == "__main__":
    main()
