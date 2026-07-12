import json
import time
from collections import Counter

from rank_bm25 import BM25Okapi

from config import DATA_DIR, FIGURES_TOP_K, TOP_K
from pipeline.retrieval import load_index, search

RESULTS_PATH = DATA_DIR / "eval_results.json"

TEST_QUERIES = [
    "Mamba model architecture diagram with selective scan mechanism",
    "Compare attention mechanisms in Mamba vs Transformer",
    "Hybrid architecture combining Mamba and attention layers",
    "Performance comparison of Mamba and Transformer on long sequences",
    "State space model formulation and discretization",
    "Training loss curves and convergence of SSM models",
    "Model scaling laws for Mamba vs Transformer",
    "Hardware-efficient implementation of selective scan",
    "Jamba architecture and its components",
    "Results of Mamba on language modeling benchmarks",
    "How does Mamba handle long-range dependencies",
    "Comparison of recurrent and convolutional representations in SSMs",
]


def item_key(meta: dict) -> str:
    """Stable identifier for a retrieved item, shared across retrieval systems."""
    modality = meta.get("modality")
    if modality == "figure":
        return f"figure:{meta.get('image_path')}"
    if modality == "transcript":
        return f"transcript:{meta.get('lecture_id')}@{meta.get('timestamp')}"
    return f"text:{meta.get('paper_id')}#{meta.get('section')}#{meta.get('chunk_index')}"


def item_text(meta: dict) -> str:
    return meta.get("text") or meta.get("caption") or ""


def unified_search(query: str) -> list[dict]:
    return search(query, k=TOP_K)


def dense_text_only_search(query: str) -> list[dict]:
    return search(query, k=TOP_K, modality="text")


def dual_pass_search(query: str) -> list[dict]:
    """Unified search plus a dedicated figure pass, deduplicated by image path.
    Mirrors the retrieval used for synthesis in main.py."""
    figure_results = search(query, k=FIGURES_TOP_K, modality="figure")
    mixed_results = search(query, k=TOP_K)
    figure_paths = {r["metadata"].get("image_path") for r in figure_results}
    merged = list(figure_results)
    for r in mixed_results:
        m = r["metadata"]
        if m.get("modality") == "figure" and m.get("image_path") in figure_paths:
            continue
        merged.append(r)
    return merged


class BM25AllText:
    """BM25 baseline over every textual representation in the corpus:
    paper text chunks, figure captions, and transcript segments. This is the
    fair text-only comparison, since captions and transcripts are text."""

    def __init__(self, metadata: list[dict]):
        self.items = [m for m in metadata if item_text(m).strip()]
        corpus = [item_text(m).lower().split() for m in self.items]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"score": float(scores[i]), "metadata": self.items[i]} for i in ranked]


def run_config(retrieve_fn) -> dict:
    per_query = []
    breakdown = Counter()
    caption_sources = Counter()
    coverage = {"figure": 0, "transcript": 0, "all_three": 0}
    for q in TEST_QUERIES:
        start = time.perf_counter()
        results = retrieve_fn(q)
        elapsed = time.perf_counter() - start

        counts = Counter(r["metadata"].get("modality") for r in results)
        breakdown.update(counts)
        for r in results:
            if r["metadata"].get("modality") == "figure":
                caption_sources[r["metadata"].get("caption_source", "unknown")] += 1
        if counts.get("figure", 0) > 0:
            coverage["figure"] += 1
        if counts.get("transcript", 0) > 0:
            coverage["transcript"] += 1
        if all(counts.get(m, 0) > 0 for m in ("text", "figure", "transcript")):
            coverage["all_three"] += 1

        per_query.append({
            "query": q,
            "text": counts.get("text", 0),
            "figure": counts.get("figure", 0),
            "transcript": counts.get("transcript", 0),
            "latency_s": round(elapsed, 4),
            "items": [item_key(r["metadata"]) for r in results],
        })

    total = sum(breakdown.values())
    n = len(TEST_QUERIES)
    latencies = [pq["latency_s"] for pq in per_query]
    return {
        "per_query": per_query,
        "total_items": total,
        "breakdown": {m: breakdown.get(m, 0) for m in ("text", "figure", "transcript")},
        "breakdown_pct": {
            m: round(breakdown.get(m, 0) / total * 100, 1) for m in ("text", "figure", "transcript")
        },
        "figure_caption_sources": dict(caption_sources),
        "coverage": {
            "queries_with_figure": coverage["figure"],
            "queries_with_transcript": coverage["transcript"],
            "queries_with_all_three": coverage["all_three"],
            "n_queries": n,
        },
        "latency_s": {
            "mean": round(sum(latencies) / n, 4),
            "min": round(min(latencies), 4),
            "max": round(max(latencies), 4),
        },
    }


def print_config(name: str, stats: dict):
    n = stats["coverage"]["n_queries"]
    print(f"\n{'=' * 95}\nCONFIG: {name}\n{'=' * 95}")
    print(f"{'Query':<70} {'Text':>5} {'Figs':>5} {'Trans':>6}")
    for pq in stats["per_query"]:
        short_q = pq["query"][:68] + ".." if len(pq["query"]) > 68 else pq["query"]
        print(f"{short_q:<70} {pq['text']:>5} {pq['figure']:>5} {pq['transcript']:>6}")
    b, p = stats["breakdown"], stats["breakdown_pct"]
    print(f"\nItems: {stats['total_items']} total | "
          f"text {b['text']} ({p['text']}%) | "
          f"figure {b['figure']} ({p['figure']}%) | "
          f"transcript {b['transcript']} ({p['transcript']}%)")
    if stats["figure_caption_sources"]:
        srcs = ", ".join(f"{k}: {v}" for k, v in sorted(stats["figure_caption_sources"].items()))
        print(f"Figure caption sources: {srcs}")
    c = stats["coverage"]
    print(f"Coverage: >=1 figure {c['queries_with_figure']}/{n} | "
          f">=1 transcript {c['queries_with_transcript']}/{n} | "
          f"all three {c['queries_with_all_three']}/{n}")
    lat = stats["latency_s"]
    print(f"Latency (s): mean {lat['mean']} | min {lat['min']} | max {lat['max']}")


def evaluate():
    _, metadata = load_index()
    corpus_counts = Counter(m.get("modality") for m in metadata)
    corpus_caption_sources = Counter(
        m.get("caption_source", "unknown") for m in metadata if m.get("modality") == "figure"
    )
    bm25 = BM25AllText(metadata)

    # Warm up the embedding model and index cache so latencies reflect
    # steady-state retrieval, not model loading.
    search(TEST_QUERIES[0], k=1)

    configs = {
        "unified": run_config(unified_search),
        "dual_pass": run_config(dual_pass_search),
        "dense_text_only": run_config(dense_text_only_search),
        "bm25_all_text": run_config(bm25.search),
    }

    print(f"Corpus: {dict(corpus_counts)} | figure caption sources: {dict(corpus_caption_sources)}")
    for name, stats in configs.items():
        print_config(name, stats)

    # Overlap between the full system and the BM25 text-only baseline:
    # which retrieved items are unique to each system?
    dual_items = {i for pq in configs["dual_pass"]["per_query"] for i in pq["items"]}
    bm25_items = {i for pq in configs["bm25_all_text"]["per_query"] for i in pq["items"]}
    comparison = {
        "dual_pass_unique_items": len(dual_items - bm25_items),
        "bm25_unique_items": len(bm25_items - dual_items),
        "shared_items": len(dual_items & bm25_items),
    }
    print(f"\n{'=' * 95}\nOVERLAP: dual_pass vs bm25_all_text (unique items across all queries)")
    print(f"  dual_pass only: {comparison['dual_pass_unique_items']} | "
          f"bm25 only: {comparison['bm25_unique_items']} | "
          f"shared: {comparison['shared_items']}")

    results = {
        "top_k": TOP_K,
        "figures_top_k": FIGURES_TOP_K,
        "n_queries": len(TEST_QUERIES),
        "corpus": {
            "modality_counts": dict(corpus_counts),
            "figure_caption_sources": dict(corpus_caption_sources),
        },
        "configs": configs,
        "dual_pass_vs_bm25_overlap": comparison,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()
