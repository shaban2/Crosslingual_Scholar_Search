# Cross-Lingual Scholar Search

Measuring the language gap and modality gap in multimodal scholarly retrieval,
using English/Indonesian as the test pair.

Solo research project by Shaban Lubanga. Builds on the pipeline of the joint
Multimodal Research Synthesizer project with Hippolyte Catteau-Verniers
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
```

## Headline findings so far

- English CLIP silos by language: the same query in EN vs ID returns almost
  disjoint top-10 lists (mean Jaccard 0.015). Indonesian queries barely reach
  English papers (3/15 queries), English queries barely reach Indonesian
  lectures (2/15).
- The gaps stack: Indonesian query -> English figure is the lowest-scoring
  cell in the similarity table (mean 0.499 vs 0.812 for same-language
  transcripts).
- Multilingual CLIP closes the language gap (EN/ID consistency rises to
  Jaccard 0.468; per-group score means equalize across query languages) but
  compresses scores and biases retrieval toward transcripts, drowning paper
  text - trading a language bias for a register bias.
- Figures are retrieved by no dense configuration in either language,
  replicating the modality-gap finding of the original project on a corpus
  2.3x larger.

Caveats: no human relevance judgments yet (planned); Indonesian transcripts
are auto-generated captions (disclosed as a corpus property).
