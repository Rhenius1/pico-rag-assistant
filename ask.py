"""Step 5 of the pipeline: retrieve chunks, then have a local LLM answer from them."""
import sys

import ollama

from search import retrieve

LLM_MODEL = "qwen2.5:7b"  # 7B: 93% vs 83% correct for llama3.2 (3B) in answer_eval.py, ~2.5x slower
TOP_K = 10  # reranked, so the answer is almost always inside the top 10 (experiment 4)

PROMPT = """You are a careful assistant answering questions about documents.
Answer ONLY using the context below. If the context does not contain the answer,
reply exactly: "I don't know based on the provided documents."
Cite the pages you used like [p.5].

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits)
    return PROMPT.format(context=context, question=question)


def reorder_edges(hits: list[dict]) -> list[dict]:
    """Put the best chunks at the start AND end of the context, the weakest in the middle.
    Models attend best to the edges of a long prompt ("lost in the middle")."""
    return hits[0::2] + hits[1::2][::-1]


def answer(question: str, hits: list[dict], model: str = LLM_MODEL) -> str:
    """Non-streaming answer from the given chunks (used by the answer-quality eval)."""
    reply = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        options={"temperature": 0},
    )
    return reply["message"]["content"].strip()


def ask(question: str, k: int = TOP_K) -> None:
    hits = retrieve(question, k)
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        options={"temperature": 0},  # 0 = least creative, most faithful to the context
        stream=True,
    )
    print(f"Question: {question}\n\nAnswer: ", end="", flush=True)
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print("\n\nSources retrieved:")
    for h in hits:
        print(f"  - {h['source']} p.{h['page']} (score {h['score']:.3f})")


if __name__ == "__main__":
    ask(" ".join(sys.argv[1:]) or "Which GPIO pin is connected to the user LED?")
