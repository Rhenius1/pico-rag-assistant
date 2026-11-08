"""Step 3 of the pipeline: turn text into vectors (embeddings)."""
from sentence_transformers import SentenceTransformer

from chunk import chunk_pages
from load import load_pdfs

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once (first call downloads ~90 MB)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one vector per text. Normalised, so dot product == cosine similarity."""
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


if __name__ == "__main__":
    # 1) What does an embedding look like?
    vec = embed_texts(["The Transformer relies on attention."])[0]
    print(f"One vector = {len(vec)} numbers. First 8: {[round(v, 3) for v in vec[:8]]}\n")

    # 2) Does 'similar meaning' really mean 'similar vector'?
    query = "How do I reset my password?"
    candidates = [
        "I forgot my login credentials.",
        "Password recovery steps for your account.",
        "The weather in Paris is sunny today.",
    ]
    q_vec, *c_vecs = embed_texts([query] + candidates)
    print(f"Query: {query!r}")
    for text, v in sorted(zip(candidates, c_vecs), key=lambda p: -similarity(q_vec, p[1])):
        print(f"  {similarity(q_vec, v):.3f}  {text}")

    # 3) Embed every chunk of the paper
    chunks = chunk_pages(load_pdfs())
    vectors = embed_texts([c["text"] for c in chunks])
    print(f"\nEmbedded {len(vectors)} chunks, each a vector of {len(vectors[0])} numbers.")
