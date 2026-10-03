from fastapi.testclient import TestClient

from grounded_rag.api import create_app
from grounded_rag.evals import evaluate

from .conftest import QUESTIONS


def test_api_answers_with_citations(engine):
    client = TestClient(create_app(engine))
    r = client.post("/ask", json={"question": "¿Es obligatorio usar casco en la obra?"})
    assert r.status_code == 200
    body = r.json()
    assert not body["abstained"] and body["citations"][0]["document"].startswith("Procedimiento")


def test_api_refuses_without_evidence(engine):
    client = TestClient(create_app(engine))
    body = client.post("/ask", json={"question": "¿Cuál es la contraseña del wifi?"}).json()
    assert body["abstained"] and body["citations"] == []


def test_api_validates_input(engine):
    client = TestClient(create_app(engine))
    assert client.post("/ask", json={"question": "?"}).status_code == 422


def test_eval_quality_gate(engine):
    report = evaluate(engine.retriever, QUESTIONS)
    assert report.recall_at_k >= 0.9, report.as_text()
    assert report.abstention_accuracy >= 0.9, report.as_text()


def test_concurrent_requests_are_safe(engine):
    from concurrent.futures import ThreadPoolExecutor

    client = TestClient(create_app(engine))
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(lambda _: client.post("/ask", json={"question": "¿Cuándo se hace el inventario?"}).status_code, range(24)))
    assert codes == [200] * 24
