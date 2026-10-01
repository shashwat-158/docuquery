from pypdf import PdfReader


def extract_pages(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with 1-based page numbers; skips empty pages."""
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of at most `size` characters,
    preferring to cut at a space so words are not split."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)  # step back by `overlap`, always move forward
    return [c for c in chunks if c]


def chunk_pdf(path: str, size: int, overlap: int) -> list[dict]:
    """PDF -> list of {page, chunk_index, text}. Chunks never cross page boundaries,
    so every chunk can be cited with an exact page number."""
    out = []
    for page_number, text in extract_pages(path):
        for index, chunk in enumerate(chunk_text(text, size, overlap)):
            out.append({"page": page_number, "chunk_index": index, "text": chunk})
    return out