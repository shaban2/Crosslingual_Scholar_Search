from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

from config import CHUNK_MAX_WORDS, DEFAULT_ENCODER, TRANSCRIPTS_DIR
from pipeline.encoders import get_text_model


def extract_video_id(youtube_url: str) -> str:
    if "v=" in youtube_url:
        return youtube_url.split("v=")[1].split("&")[0]
    if "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]
    return youtube_url


def fetch_transcript(youtube_url: str, languages: list[str] = ("en", "id")) -> list[dict]:
    video_id = extract_video_id(youtube_url)
    transcript = list(YouTubeTranscriptApi().fetch(video_id, languages=list(languages)))

    save_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    with open(save_path, "w") as f:
        for entry in transcript:
            f.write(f"[{entry.start:.1f}s] {entry.text}\n")
    print(f"  -> saved to {save_path}")

    return transcript


def segment_transcript(transcript: list[dict], max_words: int = CHUNK_MAX_WORDS) -> list[dict]:
    lecture_id = "lecture_unknown"
    segments = []
    current_text = []
    current_start = transcript[0].start if transcript else 0
    word_count = 0

    for entry in transcript:
        text = entry.text.strip()
        words = text.split()
        if word_count + len(words) > max_words and current_text:
            segments.append({
                "lecture_id": lecture_id,
                "timestamp": current_start,
                "text": " ".join(current_text),
            })
            current_text = []
            word_count = 0
            current_start = entry.start
        current_text.append(text)
        word_count += len(words)

    if current_text:
        segments.append({
            "lecture_id": lecture_id,
            "timestamp": current_start,
            "text": " ".join(current_text),
        })

    return segments


def assign_lecture_id(segments: list[dict], video_id: str) -> list[dict]:
    for seg in segments:
        seg["lecture_id"] = video_id
    return segments


def embed_segments(segments: list[dict], encoder: str = DEFAULT_ENCODER) -> list[tuple[list[float], dict]]:
    model = get_text_model(encoder)
    results = []
    texts = [s["text"] for s in segments]
    embeddings = model.encode(texts, show_progress_bar=False)
    for i, emb in enumerate(embeddings):
        results.append((emb.tolist(), segments[i]))
    return results
