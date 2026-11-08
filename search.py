"""Retrieval: find the chunks most relevant to a question (vector, keyword, or hybrid)."""
import sys
from functools import lru_cache

from embed import embed_texts
from hybrid import BM25, rank_order, rrf_scores
from index import get_collection
from rerank import rerank_scores

MODES = ("vector", "bm25", "hybrid")


@lru_cache(maxsize=1)
def _corpus() -> dict:
    """Load all chunks from Chroma once and build the BM25 index over them."""
    data = get_collection().get(include=["documents", "metadatas"])
    return {
        "ids": data["ids"],
        "texts": data["documents"],
        "metas": data["metadatas"],
        "bm25": BM25(data["documents"]),
    }


CANDIDATES = 20  # how many first-stage results the reranker re-scores


def retrieve(question: str, k: int = 4, mode: str = "hybrid", rerank: bool = True) -> list[dict]:
    """Return the k best chunks, best first. `score` is cosine similarity (vector), BM25 score
    (bm25), the fused Reciprocal Rank Fusion score (hybrid), or, with rerank=True, the
    cross-encoder's relevance score for the top CANDIDATES first-stage results."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    corpus = _corpus()
    rankings, vector_scores, bm25_scores = [], {}, {}

    if mode in ("vector", "hybrid"):
        result = get_collection().query(query_embeddings=embed_texts([question]), n_results=len(corpus["ids"]))
        position = {chunk_id: i for i, chunk_id in enumerate(corpus["ids"])}
        order = [position[chunk_id] for chunk_id in result["ids"][0]]
        vector_scores = {i: 1 - d for i, d in zip(order, result["distances"][0])}
        rankings.append(order)

    if mode in ("bm25", "hybrid"):
        scores = corpus["bm25"].scores(question)
        bm25_scores = {i: float(s) for i, s in enumerate(scores)}
        rankings.append(rank_order(scores))

    final_scores = {"vector": vector_scores, "bm25": bm25_scores}.get(mode) or rrf_scores(rankings)
    best = sorted(final_scores, key=final_scores.get, reverse=True)[: max(k, CANDIDATES) if rerank else k]
    if rerank:
        new_scores = rerank_scores(question, [corpus["texts"][i] for i in best])
        final_scores = dict(zip(best, new_scores))
        best = sorted(best, key=final_scores.get, reverse=True)[:k]
    return [
        {
            "text": corpus["texts"][i],
            "source": corpus["metas"][i]["source"],
            "page": corpus["metas"][i]["page"],
            "score": final_scores[i],
        }
        for i in best
    ]


if __name__ == "__main__":
    mode, args = "hybrid", sys.argv[1:]
    if args and args[0] in MODES:
        mode, args = args[0], args[1:]
    question = " ".join(args) or "Which GPIO pin is connected to the user LED?"
    print(f"Question: {question}  (mode: {mode} + rerank)\n")
    for rank, hit in enumerate(retrieve(question, mode=mode), start=1):
        print(f"#{rank}  score {hit['score']:.3f}  [{hit['source']} p.{hit['page']}]")
        print(hit["text"][:400].replace("\n", " "), "...\n")
