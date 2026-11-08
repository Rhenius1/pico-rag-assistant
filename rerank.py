"""Stage 2 of retrieval: a cross-encoder re-scores the candidates from hybrid search."""
from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def get_reranker(model_name: str = RERANK_MODEL) -> CrossEncoder:
    """Load a cross-encoder once (first call downloads it)."""
    return CrossEncoder(model_name)


def rerank_scores(question: str, texts: list[str], model_name: str = RERANK_MODEL) -> list[float]:
    """One relevance score per text. The model reads (question, text) together, so it can judge
    'does this passage answer this question', unlike an embedding that encodes each side alone."""
    return get_reranker(model_name).predict([(question, t) for t in texts]).tolist()
