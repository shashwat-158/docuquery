from app.ingestion import chunk_text


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1].split()  # neighbouring chunks overlap


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=200, overlap=40) == ["hello world"]