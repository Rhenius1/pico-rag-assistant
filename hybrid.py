"""Keyword search (BM25) and Reciprocal Rank Fusion, the two ingredients of hybrid search."""
import math
import re
from collections import Counter

import numpy as np


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Keeps identifiers like gpio29, i2c, 3v3_en intact."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25:
    """Okapi BM25: rare words count more (IDF), repeats saturate (k1), long docs are penalised (b)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(d)) for d in docs]
        self.doc_lens = np.array([sum(tf.values()) for tf in self.term_freqs])
        self.avg_len = self.doc_lens.mean()
        doc_freq = Counter(w for tf in self.term_freqs for w in tf)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in doc_freq.items()}

    def scores(self, query: str) -> np.ndarray:
        words = [w for w in tokenize(query) if w in self.idf]
        out = np.zeros(len(self.term_freqs))
        for i, tf in enumerate(self.term_freqs):
            norm = self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avg_len)
            out[i] = sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in words if w in tf)
        return out


def rank_order(scores: np.ndarray) -> list[int]:
    """Document indexes, best score first."""
    return list(np.argsort(-scores))


def rrf_scores(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank). Only ranks matter,
    so we never have to make BM25 scores and cosine similarities comparable."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Document indexes ordered by fused score, best first."""
    fused = rrf_scores(rankings, k)
    return sorted(fused, key=fused.get, reverse=True)
