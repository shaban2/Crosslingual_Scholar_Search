import requests

from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, OLLAMA_BASE_URL, PAPER_METADATA


SYSTEM_PROMPT = """You are a research synthesis assistant. Given a user query and a set of retrieved sources (text excerpts from papers, figure captions, and lecture transcript segments), generate a structured literature review.

Requirements:
1. Organize the review with sections: Overview, Key Findings, Cross-Modal Analysis, and Conclusion.
2. For every claim, cite the exact source in brackets, e.g. [Paper: Title] or [Figure from Title, Page X] or [Lecture: lecture_id @ Ts].
3. When a figure is relevant, describe what it shows and reference it.
4. When a lecture segment supports a point, quote or paraphrase it with a timestamp reference.
5. Be concise but thorough — synthesize across sources, don't just list them.
6. If sources disagree, highlight the contradiction explicitly.

Write the review in Markdown."""


def assemble_context(
    text_results: list[dict],
    figure_results: list[dict],
    transcript_results: list[dict],
) -> str:
    parts = ["## Retrieved Text Passages\n"]
    for r in text_results:
        m = r["metadata"]
        meta = PAPER_METADATA.get(m["paper_id"], {})
        label = f"'{meta['title']}' by {meta['authors']}" if meta else m["paper_id"]
        parts.append(f"**[From {label}, Section: {m['section']}]**\n{m['text']}\n")

    parts.append("\n## Retrieved Figures\n")
    for r in figure_results:
        m = r["metadata"]
        meta = PAPER_METADATA.get(m["paper_id"], {})
        label = f"'{meta['title']}'" if meta else m["paper_id"]
        caption = m.get("caption", "")
        if caption:
            parts.append(f"**[From {label}, Page {m['page_num']}]:** {caption}\n")
        else:
            parts.append(f"**[From {label}, Page {m['page_num']}]**\n")

    parts.append("\n## Retrieved Transcript Segments\n")
    for r in transcript_results:
        m = r["metadata"]
        ts = m.get("timestamp", 0)
        parts.append(f"**[Lecture: {m['lecture_id']} @ {ts:.1f}s]**\n{m['text']}\n")

    return "\n".join(parts)


def generate_review(query: str, context: str) -> str:
    user_prompt = f"## User Query\n{query}\n\n## Retrieved Sources\n{context}\n\nGenerate a structured literature review."

    payload = {
        "model": LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "num_predict": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
        },
    }

    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]
