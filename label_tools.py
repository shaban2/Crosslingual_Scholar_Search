"""Helper for the LLM annotation pass: print a query's pooled items with
stable row numbers, record judgments, and merge them into the label sheet."""

import csv
import json
import sys

from config import DATA_DIR

SHEET = DATA_DIR / "labels" / "label_sheet.csv"
JUDGMENTS = DATA_DIR / "labels" / "llm_labels.json"


def rows_for(qid: str) -> list[dict]:
    with open(SHEET) as f:
        return [r for r in csv.DictReader(f) if r["query_id"] == qid]


def show(qid: str):
    rows = rows_for(qid)
    print(f"{qid} | EN: {rows[0]['query_en']}")
    print(f"{qid} | ID: {rows[0]['query_id_indonesian']}")
    print("-" * 100)
    for i, r in enumerate(rows):
        tag = f"{r['modality'][:5]}/{r['language']}"
        snippet = r["snippet"][:200]
        key = r["item_key"] if r["modality"] == "figure" else ""
        print(f"[{i:02d}] {tag:<9} {snippet} {key}")


def record(qid: str, relevant: list[int]):
    rows = rows_for(qid)
    try:
        data = json.load(open(JUDGMENTS))
    except FileNotFoundError:
        data = {}
    data[qid] = {r["item_key"]: (1 if i in set(relevant) else 0) for i, r in enumerate(rows)}
    json.dump(data, open(JUDGMENTS, "w"), indent=1)
    print(f"{qid}: {len(relevant)} relevant / {len(rows)} judged; total queries done: {len(data)}")


def merge():
    data = json.load(open(JUDGMENTS))
    with open(SHEET) as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys()
    missing = 0
    for r in rows:
        label = data.get(r["query_id"], {}).get(r["item_key"])
        if label is None:
            missing += 1
        else:
            r["label"] = str(label)
    with open(SHEET, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"merged; {missing} rows without judgment")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "show":
        show(sys.argv[2])
    elif cmd == "merge":
        merge()
