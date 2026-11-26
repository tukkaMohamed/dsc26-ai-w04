
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bayyina.chunking import chunk_text
from bayyina.bm25 import BM25Index
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever


def is_relevant(hit, relevant):
    return any(
        hit.doc_id == item["doc_id"]
        and hit.start < item["end"]
        and hit.end > item["start"]
        for item in relevant
    )


def main():
    chunks = []

    contract_dir = ROOT / "data" / "contracts"

    for path in sorted(contract_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_text(path.stem, text, max_chars=500, overlap=50)
        )

    queries_path = ROOT / "data" / "queries.jsonl"

    queries = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    bm25 = BM25Index()
    bm25.add(chunks)

    embedder = HashingEmbedder()
    vectors = embedder.embed([chunk.text for chunk in chunks])

    hybrid = HybridRetriever(chunks, embedder)

    names = ["BM25 only", "Dense only", "Hybrid (RRF)"]
    found = {name: 0 for name in names}
    missed = {name: [] for name in names}

    for query_data in queries:
        query = query_data["query"]

        bm_hits = [
            next(chunk for chunk in chunks if chunk.chunk_id == chunk_id)
            for chunk_id, _ in bm25.search(query, top_k=3)
        ]

        query_vector = embedder.embed([query])[0]

        dense_order = sorted(
            range(len(chunks)),
            key=lambda i: (
                -float(vectors[i] @ query_vector),
                chunks[i].chunk_id,
            ),
        )[:3]

        dense_hits = [chunks[i] for i in dense_order]

        hybrid_hits = hybrid.search(
            query,
            top_k=3,
            candidates=50,
        )

        for name, hits in zip(
            names,
            [bm_hits, dense_hits, hybrid_hits],
        ):
            if any(
                is_relevant(hit, query_data["relevant"])
                for hit in hits
            ):
                found[name] += 1
            else:
                missed[name].append(query_data["qid"])

    print("| Retriever | recall@3 | queries missed |")
    print("|---|---:|---|")

    for name in names:
        score = found[name] / len(queries)
        missing = ", ".join(missed[name]) or "None"

        print(f"| {name} | {score:.3f} | {missing} |")


if __name__ == "__main__":
    main()
