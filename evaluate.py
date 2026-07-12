"""Cross-lingual retrieval evaluation.

Runs every (encoder x query language) combination over the same bilingual
corpus and measures, per cell:
  - modality distribution of retrieved items (text / figure / transcript)
  - content-language distribution of retrieved items (en / id)
  - coverage (queries returning >=1 figure, >=1 transcript, >=1 id item)
  - EN<->ID consistency: for each query pair, overlap between the results of
    the English query and its Indonesian translation. A language-fair system
    returns (near-)identical lists for both; the shortfall is the language gap.
  - mean retrieval latency after warm-up

Also runs a BM25 lexical baseline over all textual content. Results are
written to data/eval_results.json; every number in the paper comes from there.
"""

import json
import time
from collections import Counter

from rank_bm25 import BM25Okapi

from config import DATA_DIR, ENCODERS, TOP_K
from pipeline.retrieval import load_index, search

QUERIES_PATH = DATA_DIR / "queries.json"
RESULTS_PATH = DATA_DIR / "eval_results.json"


def load_queries() -> list[dict]:
    with open(QUERIES_PATH) as f:
        data = json.load(f)
    return data["queries"]


def item_key(meta: dict) -> str:
    modality = meta.get("modality")
    if modality == "figure":
        return f"figure:{meta.get('image_path')}"
    if modality == "transcript":
        # timestamps are full-precision on fresh fetches but 1-decimal when
        # reloaded from cached transcript files; normalize so the same
        # segment gets the same key regardless of which build produced it
        return f"transcript:{meta.get('lecture_id')}@{float(meta.get('timestamp', 0)):.1f}"
    return f"text:{meta.get('paper_id')}#{meta.get('section')}#{meta.get('chunk_index')}"


def item_text(meta: dict) -> str:
    return meta.get("text") or meta.get("caption") or ""


class BM25AllText:
    """Lexical baseline over every textual representation, both languages.
    Cross-lingual retrieval via BM25 works only through shared tokens
    (loanwords, code-switched English terms in Indonesian lectures)."""

    def __init__(self, metadata: list[dict]):
        self.items = [m for m in metadata if item_text(m).strip()]
        corpus = [item_text(m).lower().split() for m in self.items]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"score": float(scores[i]), "metadata": self.items[i]} for i in ranked]


def run_cell(queries: list[dict], lang_field: str, retrieve_fn) -> dict:
    per_query = []
    modality_counts = Counter()
    language_counts = Counter()
    coverage = Counter()
    latencies = []
    for q in queries:
        query_text = q[lang_field]
        start = time.perf_counter()
        results = retrieve_fn(query_text)
        latencies.append(time.perf_counter() - start)

        mods = Counter(r["metadata"].get("modality") for r in results)
        langs = Counter(r["metadata"].get("language") for r in results)
        modality_counts.update(mods)
        language_counts.update(langs)
        if mods.get("figure", 0) > 0:
            coverage["figure"] += 1
        if mods.get("transcript", 0) > 0:
            coverage["transcript"] += 1
        if langs.get("id", 0) > 0:
            coverage["id_content"] += 1
        if langs.get("en", 0) > 0:
            coverage["en_content"] += 1

        per_query.append({
            "query_id": q["id"],
            "query": query_text,
            "modalities": dict(mods),
            "languages": dict(langs),
            "items": [item_key(r["metadata"]) for r in results],
        })

    total = sum(modality_counts.values())
    n = len(queries)
    return {
        "per_query": per_query,
        "total_items": total,
        "modality_counts": {m: modality_counts.get(m, 0) for m in ("text", "figure", "transcript")},
        "language_counts": {l: language_counts.get(l, 0) for l in ("en", "id")},
        "coverage": {
            "queries_with_figure": coverage["figure"],
            "queries_with_transcript": coverage["transcript"],
            "queries_with_en_content": coverage["en_content"],
            "queries_with_id_content": coverage["id_content"],
            "n_queries": n,
        },
        "latency_s": {
            "mean": round(sum(latencies) / n, 4),
            "min": round(min(latencies), 4),
            "max": round(max(latencies), 4),
        },
    }


def en_id_consistency(cell_en: dict, cell_id: dict) -> dict:
    """Overlap between each EN query's results and its ID twin's results.
    Reported as mean Jaccard and mean overlap@K across query pairs."""
    jaccards, overlaps = [], []
    id_by_query = {pq["query_id"]: pq for pq in cell_id["per_query"]}
    for pq_en in cell_en["per_query"]:
        pq_id = id_by_query[pq_en["query_id"]]
        a, b = set(pq_en["items"]), set(pq_id["items"])
        inter = len(a & b)
        union = len(a | b)
        jaccards.append(inter / union if union else 0.0)
        overlaps.append(inter / max(len(a), 1))
    n = len(jaccards)
    return {
        "mean_jaccard": round(sum(jaccards) / n, 3),
        "mean_overlap_at_k": round(sum(overlaps) / n, 3),
        "per_query_jaccard": {
            pq["query_id"]: round(j, 3)
            for pq, j in zip(cell_en["per_query"], jaccards)
        },
    }


def print_cell(name: str, stats: dict):
    n = stats["coverage"]["n_queries"]
    m, l, c = stats["modality_counts"], stats["language_counts"], stats["coverage"]
    print(f"\n== {name} ==")
    print(f"  modalities: text {m['text']} | figure {m['figure']} | transcript {m['transcript']}"
          f"   languages: en {l['en']} | id {l['id']}")
    print(f"  coverage: fig {c['queries_with_figure']}/{n} | trans {c['queries_with_transcript']}/{n}"
          f" | en-content {c['queries_with_en_content']}/{n} | id-content {c['queries_with_id_content']}/{n}")
    print(f"  latency mean {stats['latency_s']['mean']}s")


def evaluate():
    queries = load_queries()
    results = {"top_k": TOP_K, "n_queries": len(queries), "encoders": {}}

    for encoder in sorted(ENCODERS):
        _, metadata = load_index(encoder)
        corpus = Counter((m.get("modality"), m.get("language")) for m in metadata)
        results.setdefault("corpus", {str(k): v for k, v in sorted(corpus.items())})

        search(queries[0]["en"], k=1, encoder=encoder)  # warm-up

        cell_en = run_cell(queries, "en", lambda q, e=encoder: search(q, k=TOP_K, encoder=e))
        cell_id = run_cell(queries, "id_", lambda q, e=encoder: search(q, k=TOP_K, encoder=e))
        consistency = en_id_consistency(cell_en, cell_id)

        results["encoders"][encoder] = {
            "en_queries": cell_en,
            "id_queries": cell_id,
            "en_id_consistency": consistency,
        }
        print_cell(f"{encoder} / EN queries", cell_en)
        print_cell(f"{encoder} / ID queries", cell_id)
        print(f"  EN<->ID consistency: Jaccard {consistency['mean_jaccard']}"
              f" | overlap@{TOP_K} {consistency['mean_overlap_at_k']}")

    # BM25 baseline over the corpus (encoder-independent; use clip's metadata)
    _, metadata = load_index("clip")
    bm25 = BM25AllText(metadata)
    bm_en = run_cell(queries, "en", bm25.search)
    bm_id = run_cell(queries, "id_", bm25.search)
    results["bm25"] = {
        "en_queries": bm_en,
        "id_queries": bm_id,
        "en_id_consistency": en_id_consistency(bm_en, bm_id),
    }
    print_cell("bm25 / EN queries", bm_en)
    print_cell("bm25 / ID queries", bm_id)
    print(f"  EN<->ID consistency: Jaccard {results['bm25']['en_id_consistency']['mean_jaccard']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()
