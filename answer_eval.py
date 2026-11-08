"""Answer-quality eval: run the FULL pipeline and check whether the generated answer is right.

Usage: uv run answer_eval.py [ollama-model]      (default model: llama3.2)

In-scope questions (same 30 as eval.py) are checked with regexes that must ALL match the answer.
Outcomes: correct / abstained ("I don't know") / wrong (confident, but missing the expected fact).
Out-of-scope trap questions should be refused; answering them counts as a hallucination.
"""
import json
import re
import sys
import time
from pathlib import Path

from ask import LLM_MODEL, answer, reorder_edges
from eval import EVAL_SET
from search import retrieve

# One list of regexes per question in EVAL_SET (same order). All must match the lower-cased answer.
ANSWER_KEYS = [
    [r"264"],                                              # SRAM
    [r"133"],                                              # clock speed
    [r"\b2 ?mb\b"],                                        # flash
    [r"\b51\b", r"\b21\b"],                                # dimensions
    [r"(gpio ?|pin )25"],                                  # LED pin
    [r"1\.8", r"5\.5"],                                    # VSYS range
    [r"bootsel"],                                          # USB mass storage mode
    [r"\b(3|three)\b"],                                    # ADC-capable GPIOs
    [r"\b(2|two)\b", r"uart", r"i2c", r"spi"],             # UART/I2C/SPI counts
    [r"12.?bit", r"500 ?ksps|500,?000|500 k"],             # ADC resolution + rate
    [r"\b(short|low|ground)\b"],                           # reset via RUN
    [r"rpi-rp2"],                                          # drive name
    [r"machine\.pin|pin class|pin\("],                     # MicroPython class
    [r"micro-?usb"],                                       # VBUS
    [r"vsys"],                                             # GPIO29
    [r"vsys", r"100"],                                     # 3V3_EN pull-up
    [r"300"],                                              # 3V3 max load
    [r"reference|power supply"],                           # ADC_VREF
    [r"1\.1"],                                             # USB version
    [r"swd|serial wire", r"usb|mass storage"],             # flash programming
    [r"main\.py"],                                         # auto-run script
    [r"\b(4|fifth|five)\b"],                               # temperature channel
    [r"pin\(\s*[\"']led[\"']\s*,\s*pin\.out\s*\)"],        # LED pin object
    [r"27", r"0\.706", r"0\.001721"],                      # temperature formula
    [r"i2c\(0", r"scl", r"sda"],                           # I2C setup
    [r"pwm\(pin\(\d+\)\)", r"duty|freq"],                  # PWM fade
    [r"9600"],                                             # UART baud
    [r"buck-?boost|smps"],                                 # power supply type
    [r"hello_dormant"],                                    # DORMANT program
    [r"lithium", r"\baa\b"],                               # batteries
]
assert len(ANSWER_KEYS) == len(EVAL_SET)

# Not covered by the documents: the right behaviour is to refuse, not to guess.
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 2018 football World Cup?",
    "How does the Pico compare to an Arduino Uno?",
    "Who is the CEO of Raspberry Pi Ltd?",  # (a Wi-Fi question was dropped: the SDK covers Pico W)
    "Who founded Raspberry Pi and when?",
]

CONFIGS = [  # (label, number of chunks, put best chunks at the edges?)
    ("k=4", 4, False),
    ("k=6", 6, False),
    ("k=10", 10, False),
    ("k=10 edges", 10, True),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace("’", "'")).lower()


def abstained(text: str) -> bool:
    t = clean(text)
    return "don't know" in t or "do not know" in t


def judge(answer_text: str, patterns: list[str]) -> str:
    if abstained(answer_text):
        return "abstained"
    t = clean(answer_text)
    return "correct" if all(re.search(p, t) for p in patterns) else "wrong"


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else LLM_MODEL
    if len(sys.argv) > 2:  # optional: only run the named configs, e.g. "k=10"
        CONFIGS[:] = [c for c in CONFIGS if c[0] in sys.argv[2:]]
    questions = [q for q, _ in EVAL_SET] + OUT_OF_SCOPE
    print(f"Model: {model} | {len(EVAL_SET)} in-scope + {len(OUT_OF_SCOPE)} out-of-scope questions\n", flush=True)

    pool = {q: retrieve(q, k=10) for q in questions}  # reranked, best first; configs take a slice
    runs: dict[str, list[dict]] = {}
    for label, k, edges in CONFIGS:
        start, rows = time.time(), []
        for i, q in enumerate(questions):
            hits = pool[q][:k]
            text = answer(q, reorder_edges(hits) if edges else hits, model)
            verdict = judge(text, ANSWER_KEYS[i]) if i < len(EVAL_SET) else ("abstained" if abstained(text) else "hallucinated")
            rows.append({"question": q, "verdict": verdict, "answer": text})
        runs[label] = rows
        print(f"finished {label} in {time.time() - start:.0f}s", flush=True)

    n = len(EVAL_SET)
    print(f"\n{'config':<12} | {'correct':>8} {'abstained':>10} {'wrong':>6} | {'refused OOS':>12}")
    for label, rows in runs.items():
        inscope = [r["verdict"] for r in rows[:n]]
        oos = [r["verdict"] for r in rows[n:]]
        print(f"{label:<12} | {inscope.count('correct'):>3}/{n} {inscope.count('correct') / n:>3.0%} "
              f"{inscope.count('abstained'):>6} {inscope.count('wrong'):>6} | {oos.count('abstained'):>6}/{len(oos)}")

    print("\nPer-question verdicts (c=correct a=abstained w=wrong) for", " | ".join(runs))
    for i, q in enumerate(questions[:n]):
        marks = " ".join(f"{runs[l][i]['verdict'][0]:>10}" for l in runs)
        print(f"  {marks}   {q[:70]}")
    print("\nOut-of-scope (a=refused, h=hallucinated):")
    for j, q in enumerate(OUT_OF_SCOPE):
        marks = " ".join(f"{runs[l][n + j]['verdict'][0]:>10}" for l in runs)
        print(f"  {marks}   {q}")

    out = Path(__file__).parent / "answer_eval_results.json"
    out.write_text(json.dumps({"model": model, "runs": runs}, indent=1, ensure_ascii=False))
    print(f"\nRaw answers saved to {out.name}")
