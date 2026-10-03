"""Retrieval and abstention evaluation: Recall@K, MRR and abstention accuracy."""

import json
from dataclasses import dataclass
from pathlib import Path

from .retriever import HybridRetriever


@dataclass
class EvalReport:
    recall_at_k: float
    mrr: float
    abstention_accuracy: float
    answerable: int
    unanswerable: int
    failures: list[str]

    def as_text(self) -> str:
        lines = [
            f"Recall@K            {self.recall_at_k:.2f}  ({self.answerable} answerable questions)",
            f"MRR                 {self.mrr:.2f}",
            f"Abstention accuracy {self.abstention_accuracy:.2f}  ({self.unanswerable} unanswerable questions)",
        ]
        lines += [f"  FAIL: {f}" for f in self.failures]
        return "\n".join(lines)


def evaluate(retriever: HybridRetriever, questions_file: str | Path, k: int = 3) -> EvalReport:
    rows = [json.loads(l) for l in Path(questions_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    hits_at_k, rr, abst_ok, failures = 0, 0.0, 0, []
    answerable = [r for r in rows if r["expected"]]
    unanswerable = [r for r in rows if not r["expected"]]

    for row in answerable:
        res = retriever.retrieve(row["q"], k)
        docs = [h.chunk.doc_id for h in res.hits]
        ranks = [i for i, d in enumerate(docs) if d in row["expected"]]
        if ranks and res.has_evidence:
            hits_at_k += 1
            rr += 1.0 / (ranks[0] + 1)
        else:
            failures.append(f"missed: {row['q']} -> {docs} (evidence={res.has_evidence})")

    for row in unanswerable:
        res = retriever.retrieve(row["q"], k)
        if not res.has_evidence:
            abst_ok += 1
        else:
            failures.append(f"should have abstained: {row['q']}")

    return EvalReport(
        recall_at_k=hits_at_k / len(answerable) if answerable else 1.0,
        mrr=rr / len(answerable) if answerable else 1.0,
        abstention_accuracy=abst_ok / len(unanswerable) if unanswerable else 1.0,
        answerable=len(answerable), unanswerable=len(unanswerable), failures=failures,
    )
