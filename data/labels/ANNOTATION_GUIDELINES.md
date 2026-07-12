# Relevance Annotation Guidelines

**Task:** Label the relevance of each retrieved item in `data/labels/label_sheet.csv`.

The file contains **1,038 retrieved items** across **15 search queries** (approximately **69 items per query**). These items represent the **union of all documents retrieved by every retrieval configuration**.

Each row includes:

- the query (in both English and Indonesian),
- the retrieved item's snippet,
- its modality,
- its language, and
- a **`label`** column to complete.

Assign:

- **1** = Relevant
- **0** = Not relevant

Judge relevance based on the **meaning and intent of the query**, **not** the language. For example, an Indonesian lecture clip may be relevant to an English query if it answers the same information need, and vice versa.

Use a **strict relevance standard**. Label an item as relevant only if it clearly satisfies the user's information need. Avoid giving partial credit for documents that are only loosely related, mention similar terms, or provide only peripheral information. A stricter labeling standard produces more meaningful evaluation metrics such as **Precision@10 (P@10)** and **nDCG**.

The task should take approximately **3–4 hours** to complete. These judgments enable proper IR evaluation rather than simple distribution analysis.

*Annotator: the author (native Indonesian speaker, fluent in English). Single-annotator judgments are disclosed as a limitation in the paper.*
