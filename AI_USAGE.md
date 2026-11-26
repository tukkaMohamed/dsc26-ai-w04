
# AI usage

## Tools I used, and for what

I used ChatGPT to help understand the task requirements, plan the retrieval components, draft implementation ideas, and identify edge cases. I used the local pytest suite to test the implementation and the evaluation script to measure recall@3 for BM25, dense retrieval, and hybrid retrieval.

## Where AI-written chunking or BM25 code broke one of the spec's rules

During implementation, I reviewed the chunking overlap logic carefully. Moving a chunk start backwards to find a word boundary can make the overlap larger than the requested value. I changed the logic to move forward to a valid word boundary instead, and checked the resulting chunks against the original text offsets and maximum chunk length.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

| Retriever | recall@3 | queries missed |
|---|---:|---|
| BM25 only | 0.917 | q12 |
| Dense only (HashingEmbedder) | 0.917 | q12 |
| Hybrid (RRF) | 0.917 | q12 |

Hybrid retrieval did not improve recall@3 on this dataset because all three approaches missed the same query, q12. A useful next step is to experiment with contextual chunk headers or a multilingual embedding model and evaluate whether they help retrieve the governing law clause.
