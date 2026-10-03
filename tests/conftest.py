from pathlib import Path

import pytest

from grounded_rag.engine import RagEngine

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs"
QUESTIONS = ROOT / "evals" / "questions.jsonl"


@pytest.fixture(scope="session")
def engine() -> RagEngine:
    return RagEngine(DOCS)
