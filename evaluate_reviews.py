"""Bounded end-to-end evaluation of retrieval-conditioned literature reviews.

Generates reviews for three representative query pairs using raw CLIP and
raw mCLIP retrieval. Evaluation is deliberately mechanical: citation
validity, retrieved-source coverage, required-section completion, and paired
review source overlap. It does not claim to measure factual quality.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import DATA_DIR, LLM_MODEL, OLLAMA_BASE_URL, PAPER_METADATA
from evaluate import item_key
from pipeline.retrieval import search

QUERY_IDS = ("q01", "q06", "q11")
ENCODERS = ("clip", "mclip")
TOP_K = 10
OUT_DIR = DATA_DIR / "review_evaluation"
REQUIRED_SECTIONS = ("Overview", "Key Findings", "Cross-Modal Analysis", "Conclusion")

SYSTEM_PROMPT = """You are a research synthesis assistant. Write a concise literature review using only the supplied sources.

Use exactly these Markdown sections: Overview, Key Findings, Cross-Modal Analysis, and Conclusion.
Cite every substantive claim with one or more supplied source identifiers such as [S1] or [S2][S4].
Do not invent citations, sources, findings, or numerical values.
If the retrieved evidence is insufficient, state that limitation explicitly.
Write in the language of the user query."""


def load_queries() -> list[dict]:
    with (DATA_DIR / "queries.json").open() as handle:
        queries = json.load(handle)["queries"]
    by_id = {query["id"]: query for query in queries}
    return [by_id[query_id] for query_id in QUERY_IDS]


def source_description(meta: dict) -> str:
    modality = meta.get("modality")
    if modality == "text":
        paper = PAPER_METADATA.get(meta.get("paper_id"), {})
        title = paper.get("title", meta.get("paper_id", "unknown paper"))
        return f"Paper: {title}; section: {meta.get('section', 'unknown')}; text: {meta.get('text', '')}"
    if modality == "figure":
        paper = PAPER_METADATA.get(meta.get("paper_id"), {})
        title = paper.get("title", meta.get("paper_id", "unknown paper"))
        return f"Figure: {title}; page: {meta.get('page_num', 'unknown')}; caption: {meta.get('caption', '')}"
    return (f"Lecture: {meta.get('lecture_id', 'unknown')}; timestamp: "
            f"{float(meta.get('timestamp', 0)):.1f}s; transcript: {meta.get('text', '')}")


def build_context(results: list[dict]) -> tuple[str, dict[str, dict]]:
    sources = {}
    lines = []
    for number, result in enumerate(results, 1):
        source_id = f"S{number}"
        meta = result["metadata"]
        sources[source_id] = {
            "item_key": item_key(meta),
            "modality": meta.get("modality"),
            "language": meta.get("language"),
            "score": result["score"],
        }
        lines.append(f"[{source_id}] {source_description(meta)}")
    return "\n\n".join(lines), sources


def generate(query: str, context: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": f"User query:\n{query}\n\nRetrieved sources:\n{context}\n\nWrite the review.",
        "stream": False,
        "options": {"num_predict": 1000, "temperature": 0, "seed": 20260718},
    }
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"]


def measure(review: str, sources: dict[str, dict]) -> dict:
    citations = [f"S{number}" for number in re.findall(r"(?:\[|\()S(\d+)(?:\]|\))", review)]
    bracket_citations = re.findall(r"\[(S\d+)\]", review)
    valid = [citation for citation in citations if citation in sources]
    invalid = [citation for citation in citations if citation not in sources]
    cited_sources = sorted(set(valid), key=lambda value: int(value[1:]))
    sections = {section: bool(re.search(rf"^(?:#+\s*)?(?:\*\*)?{re.escape(section)}(?:\*\*)?\s*$", review,
                                        flags=re.MULTILINE | re.IGNORECASE))
                for section in REQUIRED_SECTIONS}
    return {
        "citation_count": len(citations),
        "valid_citation_count": len(valid),
        "invalid_citations": invalid,
        "citation_validity": len(valid) / len(citations) if citations else 0.0,
        "bracket_format_compliance": len(bracket_citations) / len(citations) if citations else 0.0,
        "unique_sources_cited": len(cited_sources),
        "source_coverage": len(cited_sources) / len(sources) if sources else 0.0,
        "cited_source_ids": cited_sources,
        "required_sections": sections,
        "section_completion": sum(sections.values()) / len(sections),
        "word_count": len(review.split()),
    }


def paired_overlap(en_run: dict, id_run: dict) -> dict:
    en_keys = {en_run["sources"][sid]["item_key"] for sid in en_run["metrics"]["cited_source_ids"]}
    id_keys = {id_run["sources"][sid]["item_key"] for sid in id_run["metrics"]["cited_source_ids"]}
    union = en_keys | id_keys
    return {
        "cited_evidence_jaccard": len(en_keys & id_keys) / len(union) if union else 0.0,
        "en_unique_evidence": len(en_keys),
        "id_unique_evidence": len(id_keys),
    }


def qualitative_markdown(runs: list[dict]) -> str:
    selected = [run for run in runs if run["query_id"] == "q11"]
    lines = ["# Qualitative retrieval example: q11", "",
             "English: How does Mamba handle long-range dependencies?", "",
             "Indonesian: Bagaimana Mamba menangani dependensi jarak jauh?", ""]
    for run in selected:
        lines.extend([f"## {run['encoder']} / {run['language']}", "",
                      "| Rank | Modality | Content language | Score | Item |",
                      "|---:|---|---|---:|---|"])
        for rank, (source_id, source) in enumerate(list(run["sources"].items())[:5], 1):
            lines.append(f"| {rank} | {source['modality']} | {source['language']} | "
                         f"{source['score']:.3f} | `{source['item_key']}` |")
        lines.extend(["", f"Citation validity: {run['metrics']['citation_validity']:.3f}; "
                      f"source coverage: {run['metrics']['source_coverage']:.3f}.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="Recompute summary from existing reviews")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review_dir = OUT_DIR / "reviews"
    review_dir.mkdir(exist_ok=True)
    runs = []
    for query in load_queries():
        for encoder in ENCODERS:
            for language, field in (("en", "en"), ("id", "id_")):
                retrieved = search(query[field], k=TOP_K, encoder=encoder)
                context, sources = build_context(retrieved)
                review_path = review_dir / f"{query['id']}_{encoder}_{language}.md"
                if args.reuse and review_path.exists():
                    review = review_path.read_text()
                else:
                    review = generate(query[field], context)
                    review_path.write_text(review + "\n")
                runs.append({
                    "query_id": query["id"], "topic": query["topic"], "encoder": encoder,
                    "language": language, "query": query[field], "sources": sources,
                    "review_file": str(review_path.relative_to(DATA_DIR.parent)),
                    "metrics": measure(review, sources),
                })
                print(f"Completed {query['id']} {encoder} {language}")
    pairs = []
    for query_id in QUERY_IDS:
        for encoder in ENCODERS:
            en_run = next(run for run in runs if run["query_id"] == query_id and
                          run["encoder"] == encoder and run["language"] == "en")
            id_run = next(run for run in runs if run["query_id"] == query_id and
                          run["encoder"] == encoder and run["language"] == "id")
            pairs.append({"query_id": query_id, "encoder": encoder, **paired_overlap(en_run, id_run)})
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": LLM_MODEL,
        "temperature": 0,
        "seed": 20260718,
        "query_ids": list(QUERY_IDS),
        "encoders": list(ENCODERS),
        "top_k": TOP_K,
        "scope_note": "Mechanical grounding evaluation; not a human or factual-quality assessment.",
        "runs": runs,
        "paired_results": pairs,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUT_DIR / "qualitative_q11.md").write_text(qualitative_markdown(runs) + "\n")
    print(f"Wrote {OUT_DIR / 'results.json'} and qualitative_q11.md")


if __name__ == "__main__":
    main()
