# Annotation notes (LLM judge)

Conventions applied consistently across all 15 queries, with representative
borderline decisions. Judgments are binary and strict per
ANNOTATION_GUIDELINES.md; 127/999 items judged relevant (13%).

## Conventions

1. **Term mention is not relevance.** Passages that name the queried concept
   while discussing something else (e.g., "unlike attention-based models,
   our approach...") were labeled 0. This was the most common rejection.
2. **Agenda/announcement segments are 0.** Lecture segments that announce a
   topic ("today we will learn about attention and Transformers", "in this
   chapter we will dig into the attention mechanism") without teaching it
   were labeled 0, in both languages, even when they anchor the exact
   chapter that answers the query.
3. **Benchmark tables are 0** unless the query asks for a comparison the
   table directly makes (q05: CBOW-vs-skip-gram accuracy tables were 1).
4. **Figures were judged by content, not caption**, opening the image where
   needed. A figure is relevant if it depicts the asked-about artifact:
   the Transformer architecture diagram is 1 for "explain the architecture
   diagram" (q03) but 0 for "how does the attention mechanism work" (q01),
   which asks for the mechanism, not the block layout.
5. **Query-scope strictness.** Items answering a sibling question were 0:
   ImageNet training curves for "loss curves for deep *language* models"
   (q13); an MNIST LSTM loss discussion likewise; NSP-only passages for the
   MLM question (q08); segments defining long-range dependency without
   Mamba for q11.
6. **References/bibliography/author-list chunks: always 0.**
7. **ASR noise ignored.** Indonesian auto-captions were judged on intended
   meaning; garbled transcription did not disqualify an otherwise
   on-point teaching segment.
8. **Same item, different query: judged independently** (e.g., the
   Mamba-360 paradigm figure is 1 for q03, 0 for q01/q10).

## Notable corrections during the pass

- q03: the Mamba-360 "evolution of paradigms" figure was initially judged 0
  from its caption; opening the image showed it contains a Transformer
  block diagram (LayerNorm-MHSA-LayerNorm-MLP), so it was corrected to 1.

## Known biases of this annotation

- Single LLM judge; no inter-annotator agreement.
- 250-character snippets: truncated-but-promising text chunks defaulted to 0
  (strict), which may undercount relevance for text relative to transcripts.
