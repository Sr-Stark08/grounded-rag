from grounded_rag.chunking import chunk_markdown
from grounded_rag.text import tokenize
from grounded_rag.vectors import HashingEmbedder, VectorStore


def test_light_stemming_matches_word_families():
    assert tokenize("cotizaciones")[0] == tokenize("cotiza")[0]
    assert tokenize("guantes")[0] == tokenize("guante")[0]
    assert tokenize("mantención") == tokenize("mantencion")


def test_chunks_have_stable_ids_and_sections():
    md = "# Doc\n\n## Uno\nTexto uno.\n\n## Dos\nTexto dos."
    chunks = chunk_markdown("doc", md)
    assert [c.id for c in chunks] == ["doc#1", "doc#2"]
    assert [c.section for c in chunks] == ["Uno", "Dos"]
    assert chunks[0].title == "Doc"


def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(dim=64)
    a, b = e.embed(["stock de cemento", "stock de cemento"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_vector_store_returns_nearest_neighbour():
    e = HashingEmbedder(dim=64)
    store = VectorStore(e.dim)
    store.add(e.embed(["mantención de camiones", "vacaciones del personal", "stock de guantes"]))
    (best, sim), *_ = store.search(e.embed(["vacaciones"])[0], k=3)
    assert best == 1 and sim > 0


def test_retriever_finds_the_right_document(engine):
    r = engine.retriever.retrieve("¿Cuál es el stock mínimo de cemento?")
    assert r.has_evidence
    assert r.hits[0].chunk.doc_id == "bodega"


def test_retriever_refuses_unrelated_questions(engine):
    r = engine.retriever.retrieve("¿Quién ganó el último partido de fútbol?")
    assert not r.has_evidence
