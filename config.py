import os
from pathlib import Path

# This project loads multiple sentence-transformers models alongside FAISS in
# one process, which segfaults on macOS unless the duplicate OpenMP runtimes
# are limited to one thread. config is imported before any heavy library, so
# setting it here covers every entry point.
os.environ.setdefault("OMP_NUM_THREADS", "1")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
FIGURES_DIR = DATA_DIR / "figures"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# Encoder configurations. Each pairs a text model with an image model whose
# embedding spaces are aligned. `clip` is the English-only baseline;
# `mclip` swaps in a multilingual text encoder (50+ languages, incl.
# Indonesian) distilled to live in the same 512-d space as the CLIP
# image encoder.
ENCODERS = {
    "clip": {
        "text": "clip-ViT-B-32",
        "image": "clip-ViT-B-32",
        # CLIP's text encoder truncates input at 77 tokens.
        "max_words": 50,
    },
    "mclip": {
        "text": "clip-ViT-B-32-multilingual-v1",
        "image": "clip-ViT-B-32",
        "max_words": 50,
    },
}
DEFAULT_ENCODER = "clip"
EMBED_DIM = 512


def index_path(encoder: str) -> Path:
    return DATA_DIR / f"faiss_index_{encoder}.pkl"


OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2"
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.3

TOP_K = 10
FIGURES_TOP_K = 5
CHUNK_MAX_WORDS = 50

LLAVA_MODEL = "llava:7b"
LLAVA_CAPTION_PROMPT = "Describe this machine learning paper figure specifically. Identify the type (architecture diagram, bar chart, table, etc.), list the key labels and numbers, and state the main takeaway in one sentence."

# English papers: broad deep-learning topics with Indonesian lecture coverage.
PAPER_METADATA = {
    "1706.03762_attention": {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani et al.",
        "language": "en",
    },
    "1810.04805_bert": {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Jacob Devlin et al.",
        "language": "en",
    },
    "1512.03385_resnet": {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "Kaiming He et al.",
        "language": "en",
    },
    "1301.3781_word2vec": {
        "title": "Efficient Estimation of Word Representations in Vector Space",
        "authors": "Tomas Mikolov et al.",
        "language": "en",
    },
    "1409.1556_vgg": {
        "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition",
        "authors": "Karen Simonyan and Andrew Zisserman",
        "language": "en",
    },
    "2312.00752_mamba": {
        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors": "Albert Gu and Tri Dao",
        "language": "en",
    },
    "2403.19887_jamba": {
        "title": "Jamba: A Hybrid Transformer-Mamba Language Model",
        "authors": "Opher Lieber et al.",
        "language": "en",
    },
    "2404.16112_mamba360": {
        "title": "Mamba-360: Survey of State Space Models as Transformer Alternative for Long Sequence Modelling",
        "authors": "Badri Narayana Patro et al.",
        "language": "en",
    },
    "2405.21060_mamba2": {
        "title": "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality",
        "authors": "Tri Dao and Albert Gu",
        "language": "en",
    },
    "2406.07887_empirical_mamba": {
        "title": "An Empirical Study of Mamba-based Language Models",
        "authors": "Roger Waleffe et al.",
        "language": "en",
    },
}

# Lecture videos: language-tagged. Indonesian entries are added after
# transcript availability is verified (see docs/corpus_candidates.md).
LECTURE_METADATA = {
    # English lectures (first three carried over from the original corpus)
    "9dSkvxS2EB0": {"title": "Mamba paper review", "language": "en"},
    "7jlZlSxZZ1g": {"title": "Mamba vs Transformer comparison", "language": "en"},
    "uazVw7ImZiQ": {"title": "Selective state space models explained", "language": "en"},
    "eMlx5fFNoYc": {"title": "Attention in transformers, step-by-step (3Blue1Brown)", "language": "en"},
    "RQowiOF_FvQ": {"title": "Stanford CS231N Lecture 8: Attention and Transformers", "language": "en"},
    "viZrOnJclY0": {"title": "Word Embedding and Word2Vec, Clearly Explained (StatQuest)", "language": "en"},
    "ANyxBVxmdZ0": {"title": "Michigan Lecture 7: Convolutional Networks", "language": "en"},
    "knTc-NQSjKA": {"title": "Stanford CS224N: BERT and pretraining", "language": "en"},
    # Indonesian lectures (auto-generated captions; disclosed in the paper)
    "bjTM02jXgvE": {"title": "Deep Learning Ep. 10: Attention and Transformers (Risman Adnan)", "language": "id"},
    "Ib5Wum2SRhY": {"title": "Dasar-dasar attention; sekilas Transformer (Moch Arif Bijaksana)", "language": "id"},
    "zdBCbz7pmkY": {"title": "Deep Learning Ep. 6: Word Embedding (Risman Adnan)", "language": "id"},
    "YtmOw515BRQ": {"title": "Apa itu Word2vec? Word Vector untuk NLP (JCOp)", "language": "id"},
    "yuvtlMfo3_Y": {"title": "Deep Learning: RNN, LSTM, dan GRU (Telkom University)", "language": "id"},
    "ITc9C9-Y9uY": {"title": "Convolutional Neural Network (Rahmadya Trias)", "language": "id"},
    "c4krGYzu4DI": {"title": "Model transformer BERT untuk analisa sentimen (Novanto Yudistira)", "language": "id"},
    "O-tfsQPI3RE": {"title": "Neural Networks untuk Pemula: Soft Computing (Kuliah Informatika)", "language": "id"},
}
