FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    EMBED_CACHE=/app/.cache/fastembed

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so containers don't download it at start-up
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed')"

COPY app ./app
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]