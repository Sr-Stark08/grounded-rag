"""Answer generation with mandatory citations and post-generation guardrails."""

import re
from dataclasses import dataclass, field
from typing import Protocol

from .retriever import Hit, Retrieval
from .text import tokenize

REFUSAL = "No encontré información sobre eso en los documentos disponibles."
NO_EVIDENCE = "NO_EVIDENCE"

SYSTEM_PROMPT = f"""Respondes preguntas usando SOLO las fuentes entregadas entre etiquetas <source>.
Reglas:
- Cada afirmación debe terminar con la cita de su fuente entre corchetes, por ejemplo [epp#2].
- No uses conocimiento externo ni inventes cifras, fechas o nombres.
- Si las fuentes no responden la pregunta, responde exactamente: {NO_EVIDENCE}
- Responde en español, en máximo 3 oraciones."""


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaudeLLM:
    """Optional generator. Requires `pip install grounded-rag[llm]` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 400):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()


@dataclass
class Answer:
    text: str
    citations: list[Hit] = field(default_factory=list)
    abstained: bool = False
    mode: str = "extractive"          # "extractive" | "llm" | "llm_rejected"
    notes: list[str] = field(default_factory=list)


CITATION = re.compile(r"\[([a-z0-9_\-]+#\d+)\]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def validate(text: str, hits: list[Hit]) -> list[str]:
    """Returns guardrail violations for an LLM answer (empty list = valid)."""
    problems = []
    allowed = {h.chunk.id: h.chunk for h in hits}
    cited = CITATION.findall(text)
    if not cited:
        problems.append("answer has no citations")
    unknown = [c for c in cited if c not in allowed]
    if unknown:
        problems.append(f"cites sources that were not retrieved: {unknown}")
    cited_text = " ".join(allowed[c].text for c in cited if c in allowed)
    for num in NUMBER.findall(CITATION.sub("", text)):
        if num not in cited_text:
            problems.append(f"number {num} does not appear in the cited sources")
    return problems


def _extractive(question: str, hits: list[Hit]) -> Answer:
    """No-LLM mode: return the best matching sentence, verbatim, with its citation."""
    q = set(tokenize(question))
    best = None
    for h in hits:
        for sentence in re.split(r"(?<=\.)\s+", h.chunk.text):
            overlap = len(q & set(tokenize(sentence)))
            if best is None or overlap > best[0]:
                best = (overlap, sentence, h)
    _, sentence, hit = best
    return Answer(f"{sentence} [{hit.chunk.id}]", [hit], mode="extractive")


def answer(question: str, retrieval: Retrieval, llm: LLM | None = None) -> Answer:
    if not retrieval.has_evidence:
        return Answer(REFUSAL, abstained=True, notes=[retrieval.reason])
    hits = retrieval.hits
    if llm is None:
        return _extractive(question, hits)

    sources = "\n".join(f'<source id="{h.chunk.id}" title="{h.chunk.title}">{h.chunk.text}</source>' for h in hits)
    raw = llm.complete(SYSTEM_PROMPT, f"{sources}\n\nPregunta: {question}")
    if raw.strip() == NO_EVIDENCE:
        return Answer(REFUSAL, abstained=True, mode="llm", notes=["model reported no evidence"])
    problems = validate(raw, hits)
    if problems:
        # Never show an ungrounded answer: fall back to verbatim evidence.
        fallback = _extractive(question, hits)
        fallback.mode, fallback.notes = "llm_rejected", problems
        return fallback
    cited_ids = set(CITATION.findall(raw))
    return Answer(raw, [h for h in hits if h.chunk.id in cited_ids], mode="llm")
