# Reproducibility

## Environment

- Python dependencies are pinned in `requirements.txt`.
- Evaluated text encoders: `sentence-transformers/clip-ViT-B-32` and
  `sentence-transformers/clip-ViT-B-32-multilingual-v1`.
- Image encoder: `sentence-transformers/clip-ViT-B-32`.
- Review generator: Ollama `llama3.2:latest`; the bounded review evaluation
  uses temperature 0 and seed 20260718.
- Indexes use FAISS inner-product search over normalized 512-dimensional vectors.
- The versioned result JSON files record platform details, model configuration,
  query hashes, index hashes, cutoff values, generation time, and random seed.

## Commands

```bash
python3 -m pip install -r requirements.txt
python3 build_index.py --download --encoder clip
python3 build_index.py --encoder mclip
python3 evaluate.py
python3 analyze_gap.py
python3 calibrate.py
python3 score_labels.py
python3 run_revision_experiments.py --query-set all
python3 generate_revision_report.py
ollama serve
python3 evaluate_reviews.py
python3 summarize_review_evaluation.py
cd paper && tectonic paper.tex
```

The legacy result files remain frozen. Revised-query and review-evaluation
outputs are written to separate versioned directories so reruns do not replace
the original experimental record.

## Review-evaluation scope

The end-to-end check uses q01 (Transformer attention), q06 (CNN feature
extraction), and q11 (Mamba long-range dependencies), in English and
Indonesian, with raw CLIP and raw mCLIP retrieval at top-10. It measures
citation validity, retrieved-source coverage, required-section completion,
and overlap between the evidence cited in paired reviews. It does not measure
factual correctness or human-perceived review quality.
