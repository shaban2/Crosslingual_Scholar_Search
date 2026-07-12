"""Ablations over the figure embedding strategy, plus a per-modality
similarity-score analysis (the CLIP "modality gap").

Figure embeddings are recomputed from the images and captions already stored
in the main index's metadata, so no LLaVA calls are needed. Note that
disabling captioning is equivalent to the `image` mode at retrieval time:
captions only influence retrieval through the embedding.
"""

import json
import os
from collections import Counter

# Interleaving FAISS searches with CLIP image encoding segfaults on macOS
# unless both OpenMP runtimes are limited to a single thread. Must be set
# before faiss/torch are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np

from config import DATA_DIR, TOP_K
from evaluate import TEST_QUERIES
import pipeline.figure_pipeline as figure_pipeline
import pipeline.retrieval as retrieval
from pipeline.figure_pipeline import embed_figures
from pipeline.retrieval import embed_query, load_index

# Share one CLIP model instance across modules: loading a second copy
# alongside FAISS segfaults on some platforms (duplicate OpenMP runtimes).
figure_pipeline._model = retrieval._get_model()

RESULTS_PATH = DATA_DIR / "ablation_results.json"

MODES = ["hybrid", "image", "caption"]


def query_matrix() -> np.ndarray:
    vecs = np.array([embed_query(q) for q in TEST_QUERIES]).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def run_variant(vectors: np.ndarray, metadata: list[dict], queries: np.ndarray) -> dict:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    breakdown = Counter()
    queries_with_figures = 0
    for row in queries:
        _, indices = index.search(row[None, :], TOP_K)
        counts = Counter(metadata[i].get("modality") for i in indices[0] if i != -1)
        breakdown.update(counts)
        if counts.get("figure", 0) > 0:
            queries_with_figures += 1
    total = sum(breakdown.values())
    return {
        "breakdown": {m: breakdown.get(m, 0) for m in ("text", "figure", "transcript")},
        "breakdown_pct": {
            m: round(breakdown.get(m, 0) / total * 100, 1) for m in ("text", "figure", "transcript")
        },
        "queries_with_figure": queries_with_figures,
        "n_queries": len(TEST_QUERIES),
    }


def modality_gap(vectors: np.ndarray, metadata: list[dict], queries: np.ndarray) -> dict:
    """Mean/std of query-item cosine similarity per modality, over ALL items
    (not just top-K), aggregated across the test queries."""
    sims = queries @ vectors.T
    stats = {}
    for modality in ("text", "figure", "transcript"):
        cols = [i for i, m in enumerate(metadata) if m.get("modality") == modality]
        if not cols:
            continue
        vals = sims[:, cols]
        stats[modality] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "max": round(float(vals.max()), 4),
            "n_items": len(cols),
        }
    return stats


def ablate():
    index, metadata = load_index()
    vectors = index.reconstruct_n(0, index.ntotal)
    queries = query_matrix()

    fig_positions = [i for i, m in enumerate(metadata) if m.get("modality") == "figure"]
    figures = [metadata[i] for i in fig_positions]
    pos_by_path = {metadata[i].get("image_path"): i for i in fig_positions}

    variants = {}
    for mode in MODES:
        variant_vectors = vectors.copy()
        replaced = 0
        for emb, fig in embed_figures(figures, mode=mode):
            row = np.array(emb, dtype="float32")[None, :]
            faiss.normalize_L2(row)
            variant_vectors[pos_by_path[fig["image_path"]]] = row[0]
            replaced += 1
        if replaced < len(figures):
            print(f"Warning: {mode}: only {replaced}/{len(figures)} figures re-embedded")
        variants[mode] = run_variant(variant_vectors, metadata, queries)

    gap = modality_gap(vectors, metadata, queries)

    print(f"\n{'=' * 80}\nFIGURE EMBEDDING ABLATION (top-{TOP_K}, {len(TEST_QUERIES)} queries)\n{'=' * 80}")
    print(f"{'Mode':<10} {'Text':>6} {'Figs':>6} {'Trans':>6} {'Fig %':>7} {'Queries w/ fig':>15}")
    for mode, stats in variants.items():
        b = stats["breakdown"]
        print(f"{mode:<10} {b['text']:>6} {b['figure']:>6} {b['transcript']:>6} "
              f"{stats['breakdown_pct']['figure']:>6}% "
              f"{stats['queries_with_figure']:>10}/{stats['n_queries']}")
    print("(captioning disabled == `image` mode at retrieval time)")

    print(f"\nMODALITY GAP: query-item cosine similarity over all items (hybrid index)")
    print(f"{'Modality':<12} {'Mean':>8} {'Std':>8} {'Max':>8} {'Items':>7}")
    for modality, s in gap.items():
        print(f"{modality:<12} {s['mean']:>8} {s['std']:>8} {s['max']:>8} {s['n_items']:>7}")

    results = {"top_k": TOP_K, "figure_embedding_variants": variants, "modality_gap": gap}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    ablate()
