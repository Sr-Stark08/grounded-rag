"""Dense retrieval: pluggable embedders + an embedded vector database (sqlite-vec)."""

import hashlib
import math
import sqlite3
import threading
from typing import Protocol

import sqlite_vec

from .text import strip_accents


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic, offline embedder (character n-gram hashing).

    Not semantic like a neural model, but free, fast and reproducible: used for tests,
    CI and as a fallback when no model is installed.
    """

    def __init__(self, dim: int = 256, n: int = 3):
        self.dim, self.n = dim, n

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            t = f" {strip_accents(text.lower())} "
            for i in range(len(t) - self.n + 1):
                h = int(hashlib.md5(t[i : i + self.n].encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class FastEmbedEmbedder:
    """Free local neural embeddings (ONNX, runs on CPU). `pip install grounded-rag[dense]`."""

    def __init__(self, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(model_name=model)
        self.dim = len(next(iter(self._model.embed(["dim probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


class VectorStore:
    """sqlite-vec virtual table: a real vector index inside a single SQLite file."""

    def __init__(self, dim: int, path: str = ":memory:"):
        self.dim = dim
        # FastAPI serves requests from a thread pool: share one connection behind a lock.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(embedding float[{dim}] distance_metric=cosine)")

    def add(self, vectors: list[list[float]]) -> None:
        with self.lock:
            self.db.executemany(
                "INSERT INTO chunks(rowid, embedding) VALUES (?, ?)",
                [(i, sqlite_vec.serialize_float32(v)) for i, v in enumerate(vectors)],
            )
            self.db.commit()

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        """Returns (row index, cosine similarity), best first."""
        with self.lock:
            rows = self.db.execute(
                "SELECT rowid, distance FROM chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                [sqlite_vec.serialize_float32(vector), k],
            ).fetchall()
        return [(int(r), 1.0 - float(d)) for r, d in rows]
