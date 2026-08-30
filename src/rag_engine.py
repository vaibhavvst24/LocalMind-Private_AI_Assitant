"""
Retrieval-Augmented Generation engine.

Chunks documents, embeds them with a local Ollama embedding model, stores
vectors in a persistent Chroma collection on disk, and retrieves the most
relevant chunks for a query at answer time.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import chromadb
from pypdf import PdfReader

from . import config
from .llm_client import OllamaClient


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Chunking (simple recursive character splitter — no extra deps needed)
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to break on a sentence/word boundary near the end
        if end < n:
            boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary <= start + chunk_size // 2:
                boundary = text.rfind(" ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end) if overlap == 0 else end - overlap
        if start <= 0 or end == n:
            start = end
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class DocumentStore:
    def __init__(self, persist_dir: Path = config.CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.ollama = OllamaClient()

    def add_document(self, file_path: Path) -> int:
        """Chunk, embed, and store a single document. Returns #chunks added."""
        text = extract_text(file_path)
        chunks = chunk_text(text)
        if not chunks:
            return 0

        embeddings = self.ollama.embed_batch(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"source": file_path.name, "chunk_index": i} for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(self, question: str, top_k: int = config.TOP_K) -> list[dict]:
        query_embedding = self.ollama.embed(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "source": meta.get("source"), "distance": dist})
        return hits

    def list_sources(self) -> list[str]:
        data = self.collection.get()
        sources = {m.get("source") for m in data.get("metadatas", []) if m}
        return sorted(sources)

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_source(self, source_name: str):
        self.collection.delete(where={"source": source_name})


def build_context_block(hits: list[dict]) -> str:
    """Format retrieved chunks into a labeled context block for the prompt."""
    if not hits:
        return ""
    parts = []
    for h in hits:
        parts.append(f"[{h['source']}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)
