import re
from pathlib import Path

import fitz

from config import CHUNK_MAX_WORDS, DEFAULT_ENCODER
from pipeline.encoders import get_text_model


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    paper_id = Path(pdf_path).stem
    doc = fitz.open(pdf_path)
    sections = {"abstract": "", "introduction": "", "method": "", "results": "", "conclusion": "", "other": ""}
    current_section = "other"
    section_keywords = {
        "abstract": r"\babstract\b",
        "introduction": r"\bintroduction\b",
        "method": r"\b(method|approach|model architecture|proposed)\b",
        "results": r"\b(results|experiments|evaluation|performance)\b",
        "conclusion": r"\b(conclusion|discussion|future work)\b",
    }

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        lines = text.split("\n")
        for line in lines:
            line_stripped = line.strip().lower()
            for section_name, pattern in section_keywords.items():
                if re.search(pattern, line_stripped):
                    current_section = section_name
                    break
            sections[current_section] += line + "\n"

    doc.close()

    chunks = []
    for section_name, content in sections.items():
        content = content.strip()
        if not content:
            continue
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        chunk_index = 0
        for para in paragraphs:
            words = para.split()
            if len(words) < 20:
                continue
            # CLIP's text encoder truncates at 77 tokens, so paragraphs are
            # split into pieces of at most CHUNK_MAX_WORDS words. A short
            # final piece is merged into the previous chunk.
            pieces = [
                words[start:start + CHUNK_MAX_WORDS]
                for start in range(0, len(words), CHUNK_MAX_WORDS)
            ]
            if len(pieces) > 1 and len(pieces[-1]) < 20:
                pieces[-2].extend(pieces.pop())
            for piece in pieces:
                chunks.append({
                    "paper_id": paper_id,
                    "section": section_name,
                    "chunk_index": chunk_index,
                    "text": " ".join(piece),
                })
                chunk_index += 1

    return chunks


def embed_text_chunks(chunks: list[dict], encoder: str = DEFAULT_ENCODER) -> list[tuple[list[float], dict]]:
    model = get_text_model(encoder)
    results = []
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    for i, emb in enumerate(embeddings):
        results.append((emb.tolist(), chunks[i]))
    return results
