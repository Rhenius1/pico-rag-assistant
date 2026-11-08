"""Step 4 of the pipeline: store chunk vectors in Chroma (saved to ./chroma_db)."""
from pathlib import Path

import chromadb

from chunk import chunk_pages
from clean import filter_chunks, strip_running_lines
from embed import embed_texts
from load import load_pdfs

DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "papers"


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    # cosine distance matches how our embeddings are meant to be compared
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index() -> int:
    pages = strip_running_lines(load_pdfs())  # drop repeated headers/footers (experiment 5)
    chunks = filter_chunks(  # drop TOC, tiny and legal-boilerplate chunks; "history" stays opt-in
        chunk_pages(pages, strategy="structured"),  # list-aware chunking (experiment 3)
        drop=("toc", "tiny", "legal"),
    )
    # start clean so chunks from documents we removed don't linger in the DB
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed_texts([c["text"] for c in chunks]),
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks. Collection now holds {get_collection().count()} items in {DB_DIR}")
