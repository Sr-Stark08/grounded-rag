"""Hybrid retrieval (BM25 + dense, fused with Reciprocal Rank Fusion) with an evidence gate."""

from dataclasses import dataclass

from .bm25 import BM25
from .chunking import Chunk
from .text import tokenize
from .vectors import Embedder, VectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float        # fused RRF score
    bm25: float
    coverage: float     # share of query terms present in the chunk


@dataclass
class Retrieval:
    hits: list[Hit]
    has_evidence: bool
    reason: str


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, db_path: str = ":memory:",
                 rrf_k: int = 60, min_coverage: float = 0.5):
        self.chunks = chunks
        self.bm25 = BM25([f"{c.title} {c.section} {c.text}" for c in chunks])
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.min_coverage = min_coverage
        self.store = None
        if embedder is not None:
            self.store = VectorStore(embedder.dim, db_path)
            self.store.add(embedder.embed([f"{c.section}. {c.text}" for c in chunks]))

    def retrieve(self, query: str, k: int = 3) -> Retrieval:
        lexical = self.bm25.scores(query)
        rankings = [sorted(range(len(self.chunks)), key=lambda i: -lexical[i])]
        if self.store is not None:
            dense = self.store.search(self.embedder.embed([query])[0], k=min(20, len(self.chunks)))
            rankings.append([i for i, _ in dense])

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        q_terms = set(tokenize(query))
        hits = []
        for idx in sorted(fused, key=lambda i: -fused[i])[:k]:
            c = self.chunks[idx]
            c_terms = set(tokenize(f"{c.title} {c.section} {c.text}"))
            coverage = len(q_terms & c_terms) / len(q_terms) if q_terms else 0.0
            hits.append(Hit(c, fused[idx], lexical[idx], coverage))

        # Evidence gate: refuse instead of answering from weakly related text.
        if not hits or not q_terms:
            return Retrieval(hits, False, "empty query or corpus")
        best = max(hits, key=lambda h: h.coverage)
        if best.bm25 <= 0 or best.coverage < self.min_coverage:
            return Retrieval(hits, False, f"insufficient evidence (best coverage {best.coverage:.2f})")
        return Retrieval(hits, True, "ok")
