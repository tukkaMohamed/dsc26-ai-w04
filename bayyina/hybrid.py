from __future__ import annotations
import numpy as np
from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    if k < 0:
        raise ValueError("k must be non-negative")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, cid in enumerate(ranking, 1):
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *, k1: float = 1.5,
                 b: float = 0.75, reranker: Reranker | None = None):
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.embedder, self.reranker = embedder, reranker
        self.bm25 = BM25Index(k1, b)
        self.bm25.add(self.chunks)
        self.vectors = np.asarray(embedder.embed([c.text for c in self.chunks]), dtype=float)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.chunks):
            raise ValueError("embedder returned an invalid matrix shape")

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        if not query or not query.strip() or top_k <= 0 or candidates <= 0 or not self.chunks:
            return []
        lexical = [cid for cid, _ in self.bm25.search(query, candidates)]
        qv = np.asarray(self.embedder.embed([query]), dtype=float)
        if qv.ndim != 2 or qv.shape[0] != 1:
            raise ValueError("embedder must return one query vector")
        sims = self.vectors @ qv[0]
        dense_idx = sorted(range(len(self.chunks)),
                           key=lambda i: (-float(sims[i]), self.chunks[i].chunk_id))[:candidates]
        dense = [self.chunks[i].chunk_id for i in dense_idx]
        fused = rrf([lexical, dense])[:candidates]
        hits = [Hit(cid, self.by_id[cid].doc_id, self.by_id[cid].start,
                    self.by_id[cid].end, self.by_id[cid].text, score)
                for cid, score in fused]
        if self.reranker is not None and hits:
            scores = list(self.reranker(query, hits))
            if len(scores) != len(hits):
                raise ValueError("reranker must return one score per hit")
            hits = [Hit(h.chunk_id, h.doc_id, h.start, h.end, h.text, float(s))
                    for h, s in zip(hits, scores)]
            hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]
