import argparse
import sys
from types import SimpleNamespace

from config import (
    DEFAULT_ENCODER,
    ENCODERS,
    FIGURES_DIR,
    LECTURE_METADATA,
    PAPERS_DIR,
    TRANSCRIPTS_DIR,
    index_path,
)
from pipeline.text_pipeline import extract_text_from_pdf, embed_text_chunks
from pipeline.figure_pipeline import extract_figures_from_pdf, embed_figures, add_captions_to_figures
from pipeline.audio_pipeline import (
    fetch_transcript,
    segment_transcript,
    assign_lecture_id,
    embed_segments,
)
from pipeline.retrieval import build_faiss_index

PAPER_URLS = [
    ("1301.3781_word2vec", "https://arxiv.org/pdf/1301.3781"),
    ("1409.1556_vgg", "https://arxiv.org/pdf/1409.1556"),
    ("1512.03385_resnet", "https://arxiv.org/pdf/1512.03385"),
    ("1706.03762_attention", "https://arxiv.org/pdf/1706.03762"),
    ("1810.04805_bert", "https://arxiv.org/pdf/1810.04805"),
    ("2312.00752_mamba", "https://arxiv.org/pdf/2312.00752"),
    ("2403.19887_jamba", "https://arxiv.org/pdf/2403.19887"),
    ("2404.16112_mamba360", "https://arxiv.org/pdf/2404.16112"),
    ("2405.21060_mamba2", "https://arxiv.org/pdf/2405.21060"),
    ("2406.07887_empirical_mamba", "https://arxiv.org/pdf/2406.07887"),
]


def download_papers():
    import urllib.request

    for paper_id, url in PAPER_URLS:
        dest = PAPERS_DIR / f"{paper_id}.pdf"
        if dest.exists():
            continue
        print(f"Downloading {paper_id} from {url}")
        urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from papers and transcripts")
    parser.add_argument(
        "--encoder", choices=sorted(ENCODERS), default=DEFAULT_ENCODER,
        help="Embedding encoder configuration (see config.ENCODERS)"
    )
    parser.add_argument(
        "--no-caption", action="store_true",
        help="Skip LLaVA caption generation for figures"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download any missing papers from arXiv before indexing"
    )
    parser.add_argument(
        "--figure-embedding", choices=["hybrid", "image", "caption"], default="hybrid",
        help="Figure embedding strategy (used for ablations)"
    )
    parser.add_argument(
        "--index-path", default=None,
        help="Override the output index path (used for ablations)"
    )
    args = parser.parse_args()

    out_path = args.index_path or index_path(args.encoder)

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_papers()

    all_embeddings = []
    all_metadata = []

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PAPERS_DIR}")
        print("Run with --download to fetch them from arXiv.")
    for pdf_path in pdf_files:
        paper_id = pdf_path.stem
        print(f"Processing paper: {paper_id}")

        text_chunks = extract_text_from_pdf(str(pdf_path))
        text_embeddings = embed_text_chunks(text_chunks, encoder=args.encoder)
        for emb, meta in text_embeddings:
            meta["modality"] = "text"
            meta["language"] = "en"
            all_embeddings.append(emb)
            all_metadata.append(meta)
        print(f"  -> {len(text_embeddings)} text chunks embedded")

        figures = extract_figures_from_pdf(str(pdf_path))
        if figures:
            if not args.no_caption:
                print(f"  -> generating captions for {len(figures)} figures...")
                figures = add_captions_to_figures(figures)
            else:
                print(f"  -> skipping captions for {len(figures)} figures")
            figure_embeddings = embed_figures(figures, mode=args.figure_embedding, encoder=args.encoder)
            for emb, meta in figure_embeddings:
                meta["modality"] = "figure"
                meta["language"] = "en"
                all_embeddings.append(emb)
                all_metadata.append(meta)
            print(f"  -> {len(figure_embeddings)} figures embedded")
        else:
            print(f"  -> no figures found")

    for video_id, lecture in LECTURE_METADATA.items():
        language = lecture["language"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        transcript_file = TRANSCRIPTS_DIR / f"{video_id}.txt"

        if transcript_file.exists():
            print(f"Loading cached transcript: {transcript_file.name} [{language}]")
            with open(transcript_file) as f:
                lines = f.readlines()
            raw = []
            for line in lines:
                start_str, text = line.split("] ", 1)
                start = float(start_str.lstrip("[").rstrip("s"))
                raw.append(SimpleNamespace(text=text.strip(), start=start))
        else:
            try:
                print(f"Fetching transcript: {url} [{language}]")
                raw = fetch_transcript(url, languages=[language])
            except Exception as e:
                print(f"  -> Failed: {e}")
                continue

        segments = segment_transcript(raw)
        segments = assign_lecture_id(segments, video_id)
        transcript_embeddings = embed_segments(segments, encoder=args.encoder)
        for emb, meta in transcript_embeddings:
            meta["modality"] = "transcript"
            meta["language"] = language
            meta["source_url"] = url
            all_embeddings.append(emb)
            all_metadata.append(meta)
        print(f"  -> {len(transcript_embeddings)} transcript segments embedded")

    if not all_embeddings:
        print("No data was processed. Exiting.")
        sys.exit(1)

    build_faiss_index(all_embeddings, all_metadata, index_path=out_path)
    print(f"Index built ({args.encoder}) with {len(all_embeddings)} vectors total:")
    counts = {}
    for m in all_metadata:
        key = (m.get("modality", "?"), m.get("language", "?"))
        counts[key] = counts.get(key, 0) + 1
    for (mod, lang), count in sorted(counts.items()):
        print(f"  {mod} [{lang}]: {count}")


if __name__ == "__main__":
    main()
