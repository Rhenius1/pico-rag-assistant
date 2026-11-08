# Pico RAG Assistant

A fully **free, local** Retrieval-Augmented Generation (RAG) system that answers questions about the
Raspberry Pi Pico documentation, with page citations. No API keys and no cloud services: embeddings,
reranking and the LLM all run on your machine.

**Pipeline:** load PDFs → clean (strip headers/footers) → list-aware chunking → drop junk chunks →
embed (MiniLM) into Chroma → at query time hybrid search (vector + BM25, fused with Reciprocal Rank Fusion)
→ cross-encoder rerank of the top 20 → Qwen2.5 7B answers from the top 10 chunks.

**Measured on 30 test questions:** retrieval hit@10 100%, MRR 0.816; end-to-end answers 93% correct (28/30);
all 6 out-of-scope questions correctly refused.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/).

```bash
uv sync
ollama pull qwen2.5:7b        # default LLM, about 4.7 GB (a smaller option: ollama pull llama3.2)
```

Download the two source PDFs into `data/` (they are not committed):

```bash
curl -L -o data/pico-datasheet.pdf https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf
curl -L -o data/raspberry-pi-pico-python-sdk.pdf https://datasheets.raspberrypi.com/pico/raspberry-pi-pico-python-sdk.pdf
```

Any other text-based PDFs placed in `data/` work too (but the evals are written for the Pico documents).

## Usage

```bash
uv run index.py                          # build the index (rerun whenever data/ changes)
uv run ask.py "Which GPIO pin is connected to the user LED?"
uv run search.py "your question"         # inspect the retrieved chunks
uv run eval.py                           # retrieval eval: hit@k and MRR
uv run answer_eval.py qwen2.5:7b "k=10"  # end-to-end answer eval (about 10+ minutes)
uv run experiments.py chunk|hybrid|structured|rerank|junk
```

## Files

| File | Purpose |
|---|---|
| `load.py` | Read PDFs page by page, keeping source and page number |
| `clean.py` | Strip running headers/footers; filters for TOC, tiny, legal and release-history chunks |
| `chunk.py` | Recursive splitter and the list-aware (bullet-preserving) splitter |
| `embed.py` | Sentence-transformer embeddings |
| `index.py` | Build the Chroma vector index |
| `hybrid.py` | BM25 keyword search (from scratch) and Reciprocal Rank Fusion |
| `rerank.py` | Cross-encoder reranker |
| `search.py` | `retrieve()`: vector, BM25 or hybrid, then optional rerank |
| `ask.py` | Grounded prompt and answer generation through Ollama |
| `eval.py`, `answer_eval.py`, `experiments.py` | Retrieval eval, answer eval, and the experiment harness |

`main.py` is an unused placeholder created by `uv init`.

## Source documents

The Raspberry Pi Pico Datasheet and Pico-series Python SDK guide are © Raspberry Pi Ltd and licensed under
CC BY-ND 4.0; this repository does not redistribute them.
