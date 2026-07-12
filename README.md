# Multimodal Research Synthesizer

A system that ingests research papers (text), figures (images), and lecture recordings (audio transcripts) to produce a structured, cited literature review using cross-modal retrieval and a local LLM.

## Team Members

- **Shaban Lubanga** — Text & Figure Extraction Pipeline
- **Hippolyte Marc-Antoine Catteau-Verniers** — Audio Transcription & Embeddings, Retrieval & UI

## Problem

Researchers and students spend days reading papers, analyzing figures, and watching lecture recordings to write a literature review. Information is scattered across three modalities with no unified way to search or synthesize across all of them at once. This project builds a multimodal system that connects text, figures, and audio in one shared embedding space and generates a coherent review.

## Dataset

| Modality | Source | Quantity | Format |
|---|---|---|---|
| Text | arXiv | 5 papers on "Mamba vs Transformer" (2312.00752, 2403.19887, 2404.16112, 2405.21060, 2406.07887) | PDF |
| Figures | Extracted from PDFs via PyMuPDF | Varies by PDF (raster images only) | PNG/JPEG |
| Transcripts | YouTube (youtube-transcript-api) | 3 lecture recordings | TXT transcripts |
| Test queries | Manually written by team | 12 questions | in `evaluate.py` |

All data is publicly available online; `python build_index.py --download` fetches any missing papers from arXiv.

## Architecture

```
PDFs ──► Text/Figure Extraction ──┐
YouTube ──► Transcript ──────────┼──► Unified Embeddings (clip-ViT-B-32)
                                   │
                                   ▼
                            FAISS Vector Index
                                   │
                                   ▼
                     Retrieve relevant chunks across all modalities
                                   │
                                   ▼
                     Llama 3.2 (Ollama)
                                   │
                                   ▼
                          Literature Review
```

## How It Works

1. **Ingestion** — `text_pipeline.py`: PyMuPDF reads PDFs, extracts text by section, chunks paragraphs to ≤50 words (CLIP's text encoder truncates at 77 tokens)
2. **Figures** — `figure_pipeline.py`: PyMuPDF extracts embedded images, captions from PDF text or LLaVA (via Ollama), embeds with clip-ViT-B-32 (hybrid image+caption embedding)
3. **Transcripts** — `audio_pipeline.py`: `youtube-transcript-api` fetches captions, segments into timestamped ≤50-word chunks
4. **Embedding** — Encode all chunks into a shared vector space using `sentence-transformers` with `clip-ViT-B-32` (512-d vectors for text, images, and transcripts)
5. **Indexing** — `retrieval.py`: Stores vectors in FAISS for fast similarity search
6. **Retrieval** — User query → embed → search across all modalities → return top-K mixed results
7. **Synthesis** — `synthesis.py`: Feed retrieved text + figures + transcript excerpts to Llama 3.2 (via Ollama) → generate structured lit review with citations

## Tech Stack

- **PDF Processing:** PyMuPDF
- **Figure Extraction:** PyMuPDF + LLaVA 1.6 via Ollama (optional local captioning)
- **Audio Transcripts:** youtube-transcript-api
- **Embeddings:** sentence-transformers (clip-ViT-B-32, 512-d) for all modalities
- **Vector Search:** FAISS (in-memory, inner-product / cosine similarity)
- **LLM:** Llama 3.2 via Ollama (local)
- **UI:** Gradio

## Project Structure

```
Research_Synthesizer/
├── config.py                    # API keys, model names, paths
├── main.py                      # Gradio entry point
├── build_index.py               # One-time: process PDFs + videos → FAISS index
├── evaluate.py                  # Retrieval evaluation: 4 configs incl. BM25 + text-only baselines
├── ablate.py                    # Figure-embedding ablations + modality-gap analysis
├── data/
│   ├── papers/                  # PDF files
│   ├── figures/                 # Extracted images
│   ├── transcripts/             # Fetched transcripts
│   └── faiss_index.pkl          # Built FAISS index
└── pipeline/
    ├── __init__.py
    ├── text_pipeline.py         # Text extraction, chunking, embedding
    ├── figure_pipeline.py       # Image extraction, captioning, embedding
    ├── audio_pipeline.py        # Transcript fetch, segmentation, embedding
    ├── retrieval.py             # FAISS index build + multi-modal search
    └── synthesis.py             # Context assembly + LLM review generation
```

## How to Run

### Prerequisites

- **Ollama** running locally with `llama3.2` and `llava:7b` pulled:
  ```bash
  ollama pull llama3.2
  ollama pull llava:7b
  ```

### Setup

```bash
# Install pinned dependencies
pip install -r requirements.txt

# One-time: build the FAISS index from papers and videos
# (--download fetches any missing PDFs from arXiv)
python build_index.py --download

# Reproduce the evaluation (writes data/eval_results.json)
python evaluate.py

# Reproduce the ablations (writes data/ablation_results.json)
python ablate.py

# Launch the Gradio UI
python main.py
# Opens at http://localhost:7860
```

Type your research query into the Gradio interface (e.g. *"Compare attention mechanisms in Mamba vs Transformer"*) and the system returns a structured literature review with citations across text, figures, and transcripts.

The UI includes a **K slider** (number of sources to retrieve), a **figure gallery** showing retrieved figures, and a **metrics bar** with retrieval breakdown by modality. A loading spinner is shown during generation.

## Results

Evaluated with `evaluate.py` using 12 test queries covering architecture, performance, and methodology topics. Corpus: 2,081 vectors (1,884 text chunks, 30 figures, 167 transcript segments). Four configurations are compared — unified CLIP search, dual-pass (unified + dedicated K=5 figure pass), dense text-only, and BM25 over all textual content:

- **Unified search retrieves no figures at all** (0/12 queries; 7/12 return a transcript). A per-modality similarity analysis (`ablate.py`) shows why: CLIP's modality gap places every figure (mean cosine 0.57, max 0.71) below the *mean* text score (0.71).
- **The dual-pass configuration restores figure coverage to 12/12 queries** (60 figure items of 180 retrieved) at ~35 ms mean latency.
- **Caption-only figure embeddings beat the hybrid image+caption average** in the unified index (3 vs. 0 figures retrieved) — captions are text, so they sit on the text side of the gap.
- BM25 over all text surfaces only 2 figure captions and 8 transcript segments; the dual-pass system and BM25 share just 21 of ~180 unique retrieved items, so the two surface substantially different evidence.

Full numbers: `data/eval_results.json` and `data/ablation_results.json`. Caveat: these metrics measure what is retrieved, not relevance — no human relevance judgments yet.

## References

- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", 2021
- Llama 3.2: Meta AI, "The Llama 3 Herd of Models", 2024
- LLaVA: Liu et al., "Visual Instruction Tuning", 2024
- FAISS: Johnson et al., "Billion-scale Similarity Search with GPUs", 2019
- sentence-transformers: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", 2019
- Ollama: ollama.ai
- PyMuPDF: Artifex Software, pymupdf.readthedocs.io
- youtube-transcript-api: GitHub — jdepoix/youtube-transcript-api
