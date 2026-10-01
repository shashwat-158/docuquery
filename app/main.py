import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAIError

from app import config, embeddings, ingestion, rag
from app.schemas import AskRequest, AskResponse, UploadResponse
from app.vector_store import VectorStore

store = VectorStore(config.DATA_DIR, dim=config.EMBED_DIM)


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings.embed(["warm-up"])  # load the model once at startup, not on the first request
    yield


app = FastAPI(title="DocuQuery", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": store.index.ntotal}


@app.post("/documents", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if any(d["filename"] == filename for d in store.documents()):
        raise HTTPException(status_code=409, detail=f"{filename} is already indexed.")

    uploads = os.path.join(config.DATA_DIR, "uploads")
    os.makedirs(uploads, exist_ok=True)
    path = os.path.join(uploads, filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = ingestion.chunk_pdf(path, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found (scanned PDF?).")

    vectors = embeddings.embed([c["text"] for c in chunks])
    store.add(vectors, [{**c, "filename": filename} for c in chunks])
    return UploadResponse(
        filename=filename, pages=len({c["page"] for c in chunks}), chunks=len(chunks)
    )


@app.get("/documents")
def list_documents():
    return store.documents()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return rag.answer_question(store, req.question, req.top_k)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")