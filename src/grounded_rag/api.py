"""FastAPI service: POST /ask returns an answer with citations or an explicit refusal."""

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .engine import RagEngine


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    id: str
    document: str
    section: str
    text: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    mode: str
    citations: list[Citation]
    notes: list[str]


def create_app(engine: RagEngine | None = None) -> FastAPI:
    engine = engine or RagEngine(os.getenv("DOCS_DIR", "data/docs"))
    app = FastAPI(title="grounded-rag", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(engine.chunks)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        a = engine.ask(req.question)
        return AskResponse(
            answer=a.text, abstained=a.abstained, mode=a.mode, notes=a.notes,
            citations=[Citation(id=h.chunk.id, document=h.chunk.title, section=h.chunk.section, text=h.chunk.text)
                       for h in a.citations],
        )

    return app
