"""Wires corpus, retriever and generator together."""

from pathlib import Path

from .answer import LLM, Answer, answer
from .chunking import load_corpus
from .retriever import HybridRetriever
from .vectors import Embedder, HashingEmbedder


class RagEngine:
    def __init__(self, docs_dir: str | Path, embedder: Embedder | None = None, llm: LLM | None = None,
                 db_path: str = ":memory:", min_coverage: float = 0.5):
        self.chunks = load_corpus(docs_dir)
        self.retriever = HybridRetriever(self.chunks, embedder or HashingEmbedder(), db_path, min_coverage=min_coverage)
        self.llm = llm

    def ask(self, question: str, k: int = 3) -> Answer:
        return answer(question, self.retriever.retrieve(question, k), self.llm)
