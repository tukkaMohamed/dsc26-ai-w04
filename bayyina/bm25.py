from __future__ import annotations
import math
from collections import Counter
from bayyina.normalize import tokenize
from bayyina.types import Chunk

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be non-negative and b in [0, 1]")
        self.k1, self.b = float(k1), float(b)
        self.chunks = {}
        self.tokens = {}
        self.lengths = {}
        self.df = Counter()
        self.avgdl = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks, self.tokens, self.lengths, self.df = {}, {}, {}, Counter()
        for chunk in chunks:
            words = tokenize(chunk.text)
            self.chunks[chunk.chunk_id] = chunk
            self.tokens[chunk.chunk_id] = words
            self.lengths[chunk.chunk_id] = len(words)
            self.df.update(set(words))
        self.avgdl = sum(self.lengths.values()) / len(self.chunks) if self.chunks else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k <= 0 or not self.chunks:
            return []
        terms = set(tokenize(query))
        if not terms:
            return []
        n = len(self.chunks)
        output = []
        for cid, words in self.tokens.items():
            tf_counts = Counter(words)
            dl = self.lengths[cid]
            score = 0.0
            for term in terms:
                tf = tf_counts.get(term, 0)
                if not tf:
                    continue
                df = self.df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else tf + self.k1
                score += idf * tf * (self.k1 + 1) / denom
            if score > 0:
                output.append((cid, float(score)))
        output.sort(key=lambda x: (-x[1], x[0]))
        return output[:top_k]
