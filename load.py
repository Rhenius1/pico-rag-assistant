"""Step 1 of the pipeline: read every PDF in data/ and return its text, page by page."""
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"


def load_pdfs(data_dir: Path = DATA_DIR) -> list[dict]:
    """Return one dict per page: {"source": filename, "page": number, "text": page text}."""
    pages = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:  # skip empty pages (e.g. scanned images)
                pages.append({"source": pdf_path.name, "page": page_number, "text": text})
    return pages


if __name__ == "__main__":
    pages = load_pdfs()
    if not pages:
        print(f"No readable PDFs found in {DATA_DIR}. Add a text-based PDF and try again.")
    else:
        print(f"Loaded {len(pages)} pages from {len({p['source'] for p in pages})} file(s).\n")
        first = pages[0]
        print(f"--- {first['source']}, page {first['page']} ---")
        print(first["text"][:1000])
