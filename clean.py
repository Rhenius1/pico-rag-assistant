"""Remove noise before indexing: running headers/footers, table of contents, tiny fragments, boilerplate."""
import re
from collections import Counter


def _shape(line: str) -> str:
    """A line with its digits masked, so 'Applications information 23' == 'Applications information 24'."""
    return re.sub(r"\d+", "#", line).strip()


def strip_running_lines(pages: list[dict]) -> list[dict]:
    """Drop page headers (the same first line on most pages of a document) and numbered footers
    (a last line ending in a page number whose shape repeats on other pages)."""
    firsts, lasts, counts = {}, {}, Counter(p["source"] for p in pages)
    for p in pages:
        lines = p["text"].split("\n")
        firsts.setdefault(p["source"], Counter())[_shape(lines[0])] += 1
        lasts.setdefault(p["source"], Counter())[_shape(lines[-1])] += 1

    cleaned = []
    for p in pages:
        lines = p["text"].split("\n")
        if len(lines) > 1 and firsts[p["source"]][_shape(lines[0])] >= max(3, 0.3 * counts[p["source"]]):
            lines = lines[1:]
        if len(lines) > 1 and re.search(r"\d$", lines[-1]) and lasts[p["source"]][_shape(lines[-1])] >= 2:
            lines = lines[:-1]
        cleaned.append({**p, "text": "\n".join(lines).strip()})
    return [p for p in cleaned if p["text"]]


def is_toc(text: str) -> bool:
    """Table-of-contents entries are full of dot leaders: 'Heading . . . . . . . 12'."""
    return text.count(". . .") >= 3


def is_tiny(text: str, min_chars: int = 40) -> bool:
    return len(text.strip()) < min_chars


def is_legal(text: str) -> bool:
    """Long, mostly-uppercase blocks are legal disclaimers."""
    letters = [c for c in text if c.isalpha()]
    return len(text) > 300 and sum(c.isupper() for c in letters) / max(1, len(letters)) > 0.5


def is_release_history(text: str) -> bool:
    """Change-log entries: a 'Release History' heading, or an entry that starts with a date."""
    return bool(re.search(r"release history", text, re.I)
                or re.match(r"^(?:[A-Z]\.\d+\. )?\d{1,2} [A-Z][a-z]+ \d{4}", text.strip()))


FILTERS = {"toc": is_toc, "tiny": is_tiny, "legal": is_legal, "history": is_release_history}


def filter_chunks(chunks: list[dict], drop: tuple[str, ...] = ("toc", "tiny", "legal", "history")) -> list[dict]:
    """Remove chunks matching any named filter."""
    return [c for c in chunks if not any(FILTERS[name](c["text"]) for name in drop)]


def is_junk(text: str) -> bool:
    return any(f(text) for f in FILTERS.values())
