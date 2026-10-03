"""Split markdown documents into section-aware chunks with stable ids."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str          # "<doc_id>#<n>", stable across runs
    doc_id: str
    title: str       # document title (first H1)
    section: str     # nearest heading
    text: str


def chunk_markdown(doc_id: str, markdown: str, max_chars: int = 600) -> list[Chunk]:
    title = doc_id
    section = ""
    chunks: list[Chunk] = []
    buffer: list[str] = []

    def flush() -> None:
        body = " ".join(" ".join(buffer).split())
        if body:
            # split very long sections on sentence boundaries
            parts, current = [], ""
            for sentence in body.replace(". ", ".\n").split("\n"):
                if current and len(current) + len(sentence) + 1 > max_chars:
                    parts.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                parts.append(current)
            for part in parts:
                chunks.append(Chunk(f"{doc_id}#{len(chunks) + 1}", doc_id, title, section, part))
        buffer.clear()

    for line in markdown.splitlines():
        if line.startswith("# "):
            flush()
            title = line[2:].strip()
        elif line.startswith("#"):
            flush()
            section = line.lstrip("#").strip()
        elif line.strip():
            buffer.append(line.strip())
    flush()
    return chunks


def load_corpus(folder: str | Path, max_chars: int = 600) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8"), max_chars))
    return chunks
