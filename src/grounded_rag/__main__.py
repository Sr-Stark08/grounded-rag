"""CLI: python -m grounded_rag ask "pregunta" | python -m grounded_rag eval"""

import argparse
import os
import sys

from .engine import RagEngine
from .evals import evaluate


def main() -> int:
    p = argparse.ArgumentParser(prog="grounded_rag")
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--llm", action="store_true", help="generate with Claude (needs ANTHROPIC_API_KEY)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("question")
    e = sub.add_parser("eval"); e.add_argument("--questions", default="evals/questions.jsonl")
    e.add_argument("--min-recall", type=float, default=0.9)
    args = p.parse_args()

    llm = None
    if args.llm:
        from .answer import ClaudeLLM
        llm = ClaudeLLM()
    engine = RagEngine(args.docs, llm=llm)

    if args.cmd == "ask":
        ans = engine.ask(args.question)
        print(ans.text)
        for h in ans.citations:
            print(f"  [{h.chunk.id}] {h.chunk.title} › {h.chunk.section}")
        if ans.notes:
            print("  notes:", "; ".join(ans.notes))
        return 0

    report = evaluate(engine.retriever, args.questions)
    print(report.as_text())
    ok = report.recall_at_k >= args.min_recall and report.abstention_accuracy >= args.min_recall
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
