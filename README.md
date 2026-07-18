# Cross-Lingual Scholar Search

Repository: https://github.com/shaban2/Crosslingual_Scholar_Search

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21427042.svg)](https://doi.org/10.5281/zenodo.21427042)

Measuring the language gap and modality gap in multimodal scholarly retrieval,
using English/Indonesian as the test pair.

Solo research project by Shaban Lubanga. Builds on the pipeline of the 
Multimodal Research Synthesizer project
(github.com/shaban2/Multimodal-Research-Synthesizer), extended with
multi-encoder support, language-tagged metadata, a bilingual corpus, and a
cross-lingual evaluation harness.

## Research questions

1. How large is the language gap: does the same question asked in Indonesian
   vs English retrieve different scholarly content from a shared index?
2. Does the language gap compound with the modality gap (figures)?
3. How much of the gap do multilingual encoders close, and what new biases
   do they introduce?

## Corpus (4,888 vectors)

- 10 English arXiv papers (word2vec, VGG, ResNet, Transformer, BERT, Mamba
  family) -> 2,720 text chunks + 67 raster figures
- 8 English lecture videos (incl. 3Blue1Brown, Stanford CS231N/CS224N,
  Michigan, StatQuest) -> 1,146 transcript segments
- 8 Indonesian lecture videos (Machine Learning Indonesia, Telkom University,
  JCOp, and others; auto-generated captions) -> 955 transcript segments
- 15 bilingual EN/ID query pairs (data/queries.json)

## Reproduce

```bash
pip install -r requirements.txt
python build_index.py --download --encoder clip    # English CLIP baseline
python build_index.py --encoder mclip              # multilingual CLIP
python evaluate.py                                 # -> data/eval_results.json
python analyze_gap.py                              # -> data/gap_analysis.json
python run_revision_experiments.py --query-set all # versioned @5/@10/@20 runs
python generate_revision_report.py                 # reconciled JSON/README/LaTeX values
ollama serve
python evaluate_reviews.py                         # bounded end-to-end review check
python summarize_review_evaluation.py              # generated review-evaluation values
```

## Headline findings so far

- English CLIP silos by language: the same query in EN vs ID returns almost
  disjoint top-10 lists (mean Jaccard 0.019). Indonesian queries barely reach
  English papers (3/15 queries), English queries barely reach Indonesian
  lectures (2/15).
- The gaps stack: Indonesian query -> English figure is the lowest-scoring
  cell in the similarity table (mean 0.498 vs 0.812 for same-language
  transcripts).
- Multilingual CLIP closes the language gap (EN/ID consistency rises to
  Jaccard 0.474; per-group score means equalize across query languages) but
  compresses scores and biases retrieval toward transcripts, drowning paper
  text - trading a language bias for a register bias.
- Figures are retrieved by neither raw dense configuration in either language,
  replicating the modality-gap finding of the original project on a corpus
  2.3x larger.

Caveats: effectiveness uses provisional judgments from a single LLM assessor,
with bilingual human validation left as future work. Indonesian transcripts are
auto-generated captions (disclosed as a corpus property).

## Documentation

- `REPRODUCIBILITY.md` records commands, model identifiers, seeds, and scope.
- `ARTIFACTS.md` lists the released artifacts, repository, and permanent Zenodo DOI.
- `ETHICS_AND_DATA.md` documents third-party data, automated processing, and risks.
- `data/review_evaluation/` contains the bounded review outputs and qualitative example.

## License

Project-authored software is licensed under the MIT License. Original query
sets, annotations, and generated result files are licensed under CC BY 4.0.
Third-party papers, figures, transcripts, videos, and model weights remain
under their original copyrights and licenses. See `LICENSE` and
`DATA_LICENSE.md`.

<!-- REVISION_RESULTS_START -->
### Reconciled query-robustness results

Generated from `data/revision_results/summary.json` at top-10. Revised-query effectiveness is not directly comparable because the original judgment pool has incomplete coverage.

| Configuration | Original EN/ID Jaccard | Formal-ID EN/ID Jaccard | Formal-ID judgment coverage |
|---|---:|---:|---:|
| CLIP raw | 0.019 | 0.000 | 58.7% |
| mCLIP raw | 0.474 | 0.409 | 81.3% |
| BM25 | 0.071 | 0.011 | 52.7% |
| CLIP calibrated | 0.083 | 0.071 | 70.7% |
| mCLIP calibrated | 0.505 | 0.489 | 85.3% |
<!-- REVISION_RESULTS_END -->
