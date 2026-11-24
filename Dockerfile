FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ffmpeg lets Whisper decode mp3/m4a/webm uploads
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY vaani ./vaani
COPY frontend ./frontend
COPY data ./data

# Runtime state, the vector store and the Whisper cache live in volumes (see docker-compose.yml)
RUN useradd --create-home vaani && mkdir -p /app/.state /app/chroma_db && chown -R vaani /app
USER vaani

EXPOSE 8000 8501

# Default: the API. docker-compose overrides this for the UI service.
CMD ["uvicorn", "vaani.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
