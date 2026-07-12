import base64
from pathlib import Path

import fitz
import requests
from PIL import Image

from config import DEFAULT_ENCODER, FIGURES_DIR, LLAVA_CAPTION_PROMPT, LLAVA_MODEL, OLLAMA_BASE_URL
from pipeline.encoders import get_image_model, get_text_model


def _get_page_caption(page) -> str:
    text = page.get_text()
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("figure ") or stripped.lower().startswith("fig. "):
            return stripped
    return ""


def extract_figures_from_pdf(pdf_path: str) -> list[dict]:
    paper_id = Path(pdf_path).stem
    doc = fitz.open(pdf_path)
    figures = []
    paper_fig_dir = FIGURES_DIR / paper_id
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]
        caption = _get_page_caption(page)
        images = page.get_images(full=True)
        for fig_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            filename = f"page{page_num+1}_fig{fig_idx+1}.{ext}"
            filepath = paper_fig_dir / filename
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            figures.append({
                "paper_id": paper_id,
                "page_num": page_num + 1,
                "figure_index": fig_idx + 1,
                "image_path": str(filepath),
                "caption": caption,
                "caption_source": "pdf" if caption else "none",
            })

    doc.close()
    return figures


def embed_figures(
    figures: list[dict], mode: str = "hybrid", encoder: str = DEFAULT_ENCODER
) -> list[tuple[list[float], dict]]:
    """Embed figures as `hybrid` (average of image and caption embeddings),
    `image` (image embedding only), or `caption` (caption embedding only,
    falling back to the image when a figure has no caption).

    Captions use the encoder's text model and images its image model; the
    two are alignment-paired in config.ENCODERS, so averaging stays valid."""
    text_model = get_text_model(encoder)
    image_model = get_image_model(encoder)
    results = []
    for fig in figures:
        try:
            image = Image.open(fig["image_path"]).convert("RGB")
            caption = fig.get("caption", "")
            has_caption = bool(caption and caption.strip())
            if mode == "caption" and has_caption:
                emb = text_model.encode(caption).tolist()
            elif mode == "hybrid" and has_caption:
                img_emb = image_model.encode(image)
                text_emb = text_model.encode(caption)
                emb = ((img_emb + text_emb) / 2).tolist()
            else:
                emb = image_model.encode(image).tolist()
            results.append((emb, fig))
        except Exception as e:
            print(f"Warning: could not embed {fig['image_path']}: {e}")
    return results


def generate_figure_caption(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": LLAVA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": LLAVA_CAPTION_PROMPT,
                    "images": [b64],
                }
            ],
            "stream": False,
            "options": {"num_predict": 256, "temperature": 0.2},
        }

        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"Warning: could not caption {image_path}: {e}")
        return None


def add_captions_to_figures(figures: list[dict]) -> list[dict]:
    """LLaVA-caption figures lacking a PDF caption. Generated captions are
    cached to disk so every index build uses identical caption text."""
    import json

    cache_path = FIGURES_DIR / "caption_cache.json"
    try:
        cache = json.load(open(cache_path))
    except FileNotFoundError:
        cache = {}

    total = len(figures)
    for i, fig in enumerate(figures):
        if fig.get("caption", "").strip():
            continue
        if fig["image_path"] in cache:
            fig["caption"] = cache[fig["image_path"]]
            fig["caption_source"] = "llava"
            continue
        caption = generate_figure_caption(fig["image_path"])
        fig["caption"] = caption or ""
        fig["caption_source"] = "llava" if caption else "none"
        if caption:
            cache[fig["image_path"]] = caption
            json.dump(cache, open(cache_path, "w"), indent=1)
        print(f"  -> captioned figure {i+1}/{total}")
    return figures
