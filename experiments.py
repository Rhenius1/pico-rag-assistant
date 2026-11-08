"""Compare pipeline settings with the same eval set. Each run rebuilds the index in memory."""
import sys

import numpy as np

from chunk import chunk_pages
from embed import embed_texts
from eval import EVAL_SET, normalise
from hybrid import BM25, rank_order, rrf
from load import load_pdfs

PAGES = load_pdfs()
QUESTIONS = [q for q, _ in EVAL_SET]
QUESTION_VECS = np.array(embed_texts(QUESTIONS))


def rankings_for(chunks: list[dict]) -> dict[str, list[list[int]]]:
    """For every question, the full ranking of chunk indexes under each retrieval mode."""
    vecs = np.array(embed_texts([c["text"] for c in chunks]))
    bm25 = BM25([c["text"] for c in chunks])
    out = {"vector": [], "bm25": [], "hybrid": []}
    for question, qvec in zip(QUESTIONS, QUESTION_VECS):
        vec_rank = rank_order(vecs @ qvec)
        bm25_rank = rank_order(bm25.scores(question))
        out["vector"].append(vec_rank)
        out["bm25"].append(bm25_rank)
        out["hybrid"].append(rrf([vec_rank, bm25_rank]))
    return out


def first_hit_ranks(chunks: list[dict], rankings: list[list[int]]) -> list[int | None]:
    texts = [normalise(c["text"]) for c in chunks]
    ranks = []
    for order, (_, phrases) in zip(rankings, EVAL_SET):
        ranks.append(next((r for r, i in enumerate(order[:10], 1) if any(p in texts[i] for p in phrases)), None))
    return ranks


def summarise(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        **{f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in (1, 4, 6, 10)},
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def show(label: str, chunks: int, s: dict) -> None:
    print(f"{label:<22} {chunks:>6} | {s['hit@1']:>6.0%} {s['hit@4']:>6.0%} {s['hit@6']:>6.0%} {s['hit@10']:>6.0%} | {s['mrr']:.3f}")


def header(first: str) -> None:
    print(f"{first:<22} {'chunks':>6} | {'hit@1':>6} {'hit@4':>6} {'hit@6':>6} {'hit@10':>6} | MRR")


def experiment_chunk_size() -> None:
    print("\nExperiment 1a: chunk size, vector search only (overlap ~19% of size)")
    header("size / overlap")
    for size in (300, 500, 800, 1200, 1600):
        chunks = chunk_pages(PAGES, size, round(size * 0.19))
        show(f"{size} / {round(size * 0.19)}", len(chunks), summarise(first_hit_ranks(chunks, rankings_for(chunks)["vector"])))


def experiment_hybrid() -> None:
    print("\nExperiment 2: retrieval mode x chunk size")
    header("mode @ size")
    for size in (800, 1200):
        chunks = chunk_pages(PAGES, size, round(size * 0.19))
        all_rankings = rankings_for(chunks)
        results = {m: first_hit_ranks(chunks, r) for m, r in all_rankings.items()}
        for mode, ranks in results.items():
            show(f"{mode} @ {size}", len(chunks), summarise(ranks))
        if size == 800:
            print("\n  Per-question change, vector -> hybrid (size 800):")
            for q, v, h in zip(QUESTIONS, results["vector"], results["hybrid"]):
                if v != h:
                    print(f"    {str(v or 'miss'):>4} -> {str(h or 'miss'):<4} {q}")
            print()


def experiment_structured() -> None:
    print("\nExperiment 3: list-aware (structured) chunking vs recursive")
    header("strategy @ size / mode")
    configs = [
        ("recursive+ov150", "recursive", 800, 152), ("recursive+ov0", "recursive", 800, 0),
        ("structured", "structured", 800, 0),
        ("recursive+ov150", "recursive", 1200, 228), ("recursive+ov0", "recursive", 1200, 0),
        ("structured", "structured", 1200, 0),
    ]
    tracked = {}
    for label, strategy, size, overlap in configs:
        chunks = chunk_pages(PAGES, size, overlap, strategy)
        all_rankings = rankings_for(chunks)
        for mode in ("vector", "hybrid"):
            ranks = first_hit_ranks(chunks, all_rankings[mode])
            show(f"{label} @{size} {mode}", len(chunks), summarise(ranks))
            tracked[(label, size, mode)] = ranks
        print()
    print("  Ranks of the two hard questions (vector / hybrid), size 800:")
    for qi in (1, 9):
        row = "  ".join(f"{label}: {tracked[(label, 800, 'vector')][qi] or 'miss'}/{tracked[(label, 800, 'hybrid')][qi] or 'miss'}"
                        for label in ("recursive+ov150", "structured"))
        print(f"    {QUESTIONS[qi][:50]:<50} {row}")


def experiment_rerank() -> None:
    import time

    from rerank import get_reranker

    print("\nExperiment 4: cross-encoder reranking of hybrid candidates (structured chunks @800)")
    chunks = chunk_pages(PAGES, 800, 0, "structured")
    hybrid = rankings_for(chunks)["hybrid"]
    pool = 30
    header("model / candidates")
    show("hybrid (no rerank)", len(chunks), summarise(first_hit_ranks(chunks, hybrid)))
    best = None
    for model in ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        encoder = get_reranker(model)
        start = time.time()
        scores = [encoder.predict([(q, chunks[i]["text"]) for i in order[:pool]]) for q, order in zip(QUESTIONS, hybrid)]
        per_query = (time.time() - start) / len(QUESTIONS)
        for n in (10, 20, 30):
            reranked = [[order[j] for j in np.argsort(-sc[:n])] for order, sc in zip(hybrid, scores)]
            ranks = first_hit_ranks(chunks, reranked)
            summary = summarise(ranks)
            show(f"{model.split('-')[-3] + '-' + model.split('-')[-2]} / top {n}", len(chunks), summary)
            if best is None or summary["mrr"] > best[0]:
                best = (summary["mrr"], model, n, ranks)
        print(f"  ({model.split('/')[1]}: {per_query * 1000:.0f} ms per query for {pool} candidates)\n")

    _, model, n, ranks = best
    before = first_hit_ranks(chunks, hybrid)
    print(f"Best: {model} / top {n}. Per-question change, hybrid -> reranked:")
    for q, b, a in zip(QUESTIONS, before, ranks):
        if b != a:
            print(f"    {str(b or 'miss'):>4} -> {str(a or 'miss'):<4} {q}")


def reranked_order(chunks: list[dict], hybrid: list[list[int]], pool: int = 20) -> list[list[int]]:
    """Re-order each question's top `pool` hybrid candidates with the cross-encoder."""
    from rerank import get_reranker

    encoder = get_reranker()
    out = []
    for question, order in zip(QUESTIONS, hybrid):
        scores = encoder.predict([(question, chunks[i]["text"]) for i in order[:pool]])
        out.append([order[j] for j in np.argsort(-scores)])
    return out


def experiment_junk() -> None:
    from clean import filter_chunks, is_junk, strip_running_lines

    print("\nExperiment 5: junk filtering (structured chunks @800, hybrid -> rerank top 20)")
    configs = [  # (label, strip headers/footers?, chunk filters)
        ("baseline (no cleaning)", False, ()),
        ("drop TOC", False, ("toc",)),
        ("strip headers/footers", True, ()),
        ("drop tiny", False, ("tiny",)),
        ("core: TOC+strip+tiny", True, ("toc", "tiny")),
        ("core + legal", True, ("toc", "tiny", "legal")),
        ("all (+ release history)", True, ("toc", "tiny", "legal", "history")),
    ]
    header("config")
    print(f"{'':<24}{'':>6}   (junk chunks among the final top 10, averaged per question, in brackets)")
    for label, strip, drops in configs:
        pages = strip_running_lines(PAGES) if strip else PAGES
        chunks = filter_chunks(chunk_pages(pages, 800, 0, "structured"), drops) if drops else chunk_pages(pages, 800, 0, "structured")
        hybrid = rankings_for(chunks)["hybrid"]
        final = reranked_order(chunks, hybrid)
        junk_in_top10 = sum(is_junk(chunks[i]["text"]) for order in final for i in order[:10]) / len(final)
        show(f"{label} (hyb)", len(chunks), summarise(first_hit_ranks(chunks, hybrid)))
        show(f"{label} (+rr) [{junk_in_top10:.1f}]", len(chunks), summarise(first_hit_ranks(chunks, final)))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("chunk", "all"):
        experiment_chunk_size()
    if which in ("hybrid", "all"):
        experiment_hybrid()
    if which in ("structured", "all"):
        experiment_structured()
    if which in ("rerank", "all"):
        experiment_rerank()
    if which in ("junk", "all"):
        experiment_junk()
