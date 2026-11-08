"""Step 2 of the pipeline: split page text into overlapping chunks."""
from load import load_pdfs

CHUNK_SIZE = 800     # target max characters per chunk
CHUNK_OVERLAP = 150  # characters shared between neighbouring chunks
SEPARATORS = ["\n\n", "\n", ". ", " "]  # try the biggest natural break first


def split_text(text: str, size: int = CHUNK_SIZE, separators: list[str] = SEPARATORS) -> list[str]:
    """Recursively split text at the largest separator so every piece is <= size."""
    if len(text) <= size:
        return [text]
    if not separators:  # no separator left: hard cut
        return [text[i:i + size] for i in range(0, len(text), size)]

    sep, rest = separators[0], separators[1:]
    pieces = []
    for part in text.split(sep):
        pieces.extend(split_text(part, size, rest) if len(part) > size else [part])

    # merge small neighbouring pieces back together up to `size`
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def add_overlap(chunks: list[str], overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Prefix each chunk with the tail of the previous one."""
    if overlap <= 0:
        return chunks
    return [chunks[0]] + [prev[-overlap:] + " " + cur for prev, cur in zip(chunks, chunks[1:])]


BULLETS = ("•", "◦")


def is_bullet(line: str) -> bool:
    return line.lstrip().startswith(BULLETS)


def split_sections(text: str) -> list[tuple[str, str, list[str]]]:
    """Cut a page into ("prose", text, []) and ("list", lead_in, items) sections.
    A list is a run of bullet lines plus the non-bullet line right before it (its lead-in).
    Each item is a top-level bullet with its nested sub-bullets attached."""
    lines = text.split("\n")
    sections, prose, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        has_lead_in = not is_bullet(line) and i + 1 < len(lines) and is_bullet(lines[i + 1])
        if not (has_lead_in or is_bullet(line)):
            prose.append(line)
            i += 1
            continue
        if prose:
            sections.append(("prose", "\n".join(prose), []))
            prose = []
        j = i + 1 if has_lead_in else i
        items = []
        while j < len(lines) and is_bullet(lines[j]):
            if lines[j].lstrip().startswith("•") or not items:
                items.append(lines[j])
            else:  # a "◦" sub-bullet belongs to the item above it
                items[-1] += "\n" + lines[j]
            j += 1
        sections.append(("list", line if has_lead_in else "", items))
        i = j
    if prose:
        sections.append(("prose", "\n".join(prose), []))
    return sections


def split_structured(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """List-aware chunking: keep each bullet list together with its lead-in; split long lists
    between items (repeating the lead-in); use the recursive splitter for ordinary prose."""
    chunks = []
    for kind, lead, items in split_sections(text):
        if kind == "prose":
            chunks.extend(c for c in split_text(lead, size) if c.strip())
            continue
        current = []
        for item in items:
            candidate = "\n".join(([lead] if lead else []) + current + [item])
            if current and len(candidate) > size:
                chunks.append("\n".join(([lead] if lead else []) + current))
                current = []
            current.append(item)
        chunks.append("\n".join(([lead] if lead else []) + current))
    return chunks


def chunk_pages(pages: list[dict], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                strategy: str = "recursive") -> list[dict]:
    """Turn pages into chunks, keeping source + page so we can cite them later.
    strategy "recursive" = split by separators + overlap; "structured" = list-aware, no overlap."""
    result = []
    for page in pages:
        if strategy == "structured":
            pieces = split_structured(page["text"], size)
        else:
            pieces = add_overlap(split_text(page["text"], size), overlap)
        for i, piece in enumerate(pieces):
            result.append({
                "id": f"{page['source']}-p{page['page']}-c{i}",
                "source": page["source"],
                "page": page["page"],
                "text": piece,
            })
    return result


if __name__ == "__main__":
    chunks = chunk_pages(load_pdfs())
    sizes = [len(c["text"]) for c in chunks]
    print(f"{len(chunks)} chunks | min {min(sizes)} | avg {sum(sizes) // len(sizes)} | max {max(sizes)} chars\n")
    for c in chunks[3:6]:
        print(f"=== {c['id']} ({len(c['text'])} chars) ===")
        print(c["text"], "\n")
