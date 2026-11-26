
# Retrieval evaluation

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`, `HashingEmbedder()`, and the 12 queries in `data/queries.jsonl`.

A hit is relevant when its span overlaps one of the query's relevant ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---:|---|
| BM25 only | 0.917 | q12 |
| Dense only (HashingEmbedder) | 0.917 | q12 |
| Hybrid (RRF) | 0.917 | q12 |

## Analysis

All three retrieval approaches achieved a recall@3 of 0.917, retrieving a relevant result for 11 out of 12 queries. The only missed query was q12, which asks about the governing law and competent court for the agency agreement. Hybrid retrieval did not improve recall over either individual retriever on this evaluation set.

A useful next experiment would be to add contextual chunk headers containing the document title and clause heading, then compare recall@3 again. A multilingual embedding model or a reranker could also be tested to improve retrieval of legal clauses with different wording.
