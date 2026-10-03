from grounded_rag.answer import NO_EVIDENCE, REFUSAL, answer


class FakeLLM:
    def __init__(self, reply: str):
        self.reply, self.calls = reply, 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


Q = "¿Cada cuánto se reponen los guantes?"


def test_extractive_mode_quotes_evidence_with_citation(engine):
    a = engine.ask(Q)
    assert "15 días" in a.text and "[epp#" in a.text
    assert not a.abstained and a.mode == "extractive"


def test_valid_llm_answer_is_accepted(engine):
    retrieval = engine.retriever.retrieve(Q)
    cid = next(h.chunk.id for h in retrieval.hits if "guantes se reponen" in h.chunk.text)
    a = answer(Q, retrieval, FakeLLM(f"Se reponen cada 15 días [{cid}]."))
    assert a.mode == "llm" and a.citations[0].chunk.id == cid


def test_answer_citing_unknown_source_is_rejected(engine):
    a = answer(Q, engine.retriever.retrieve(Q), FakeLLM("Cada 15 días [rrhh#9]."))
    assert a.mode == "llm_rejected"
    assert any("not retrieved" in n for n in a.notes)


def test_answer_without_citations_is_rejected(engine):
    a = answer(Q, engine.retriever.retrieve(Q), FakeLLM("Cada 15 días."))
    assert a.mode == "llm_rejected"


def test_invented_number_is_rejected(engine):
    retrieval = engine.retriever.retrieve(Q)
    cid = retrieval.hits[0].chunk.id
    a = answer(Q, retrieval, FakeLLM(f"Se reponen cada 30 días [{cid}]."))
    assert a.mode == "llm_rejected"
    assert any("30" in n for n in a.notes)


def test_model_reporting_no_evidence_abstains(engine):
    a = answer(Q, engine.retriever.retrieve(Q), FakeLLM(NO_EVIDENCE))
    assert a.abstained and a.text == REFUSAL


def test_llm_is_never_called_without_evidence(engine):
    llm = FakeLLM("cualquier cosa")
    a = answer("¿Cuál es el precio del dólar hoy?", engine.retriever.retrieve("¿Cuál es el precio del dólar hoy?"), llm)
    assert a.abstained and llm.calls == 0
