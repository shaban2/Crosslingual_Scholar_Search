# arXiv submission metadata

## Title

Lost in One Index: Measuring How Language and Modality Gaps Compound in
English--Indonesian Multimodal Scholarly Search

## Author

Shaban Lubanga

Department of Information Systems, Institut Teknologi Sepuluh Nopember,
Surabaya, Indonesia

## Categories

- Primary: `cs.IR` (Information Retrieval)
- Suggested cross-lists: `cs.CL` (Computation and Language), `cs.MM`
  (Multimedia)

## Comments

6 pages, 5 tables. Code and reproducibility artifacts are available at
https://github.com/shaban2/Crosslingual_Scholar_Search and archived under the
permanent DOI https://doi.org/10.5281/zenodo.21427042.

## License

Suggested arXiv license: Creative Commons Attribution 4.0 (CC BY 4.0). The
author must confirm this choice in the arXiv submission form.

## Abstract

Scholarly knowledge in Indonesia is distributed across languages. Research
papers are commonly published in English, whereas lectures and theses are
often produced in Indonesian. Retrieval systems that embed these sources in
a single vector index could help connect them, but their cross-lingual
behavior has received limited evaluation. We construct a bilingual,
multimodal test collection containing ten English papers (2,720 text chunks
and 67 figures), eight English and eight Indonesian lecture videos (2,101
transcript segments), and 15 translation-paired queries. We then measure how
retrieval changes when the same question is expressed in English or
Indonesian. With an English-only CLIP encoder, paired queries produce nearly
disjoint top-10 lists (mean Jaccard overlap 0.019). Indonesian queries
retrieve English papers in only 3 of 15 cases. The language and modality gaps
also interact, with Indonesian-query-to-figure similarity forming the
lowest-scoring group in the collection (mean 0.498, compared with 0.812 for
same-language transcripts). A multilingual CLIP text encoder improves
consistency substantially (Jaccard 0.474), but favors lecture transcripts
over paper text. Per-group score calibration improves coverage. Figures are
retrieved for up to 7 of 15 queries, and every Indonesian query retrieves an
English paper. However, calibration reduces Indonesian-query effectiveness
(nDCG@10 0.120 to 0.051), indicating a fairness-precision tradeoff. Under
provisional judgments from a single LLM assessor, BM25 outperforms every
dense configuration in both languages, partly because Indonesian lectures
contain code-switched English terminology. A separately versioned formal
Indonesian query set also shows that retrieval is sensitive to register,
although mCLIP is more stable than CLIP or BM25. We release the collection,
provisional judgments, query audit, and code.

## Source archive

Upload `arxiv-submission.tar.gz`. Its root contains only:

- `paper.tex`
- `generated_results.tex`
- `generated_review_results.tex`

Select LaTeX/PDFLaTeX processing in arXiv and inspect the generated PDF before
final submission.
