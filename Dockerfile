# Audio Transcriber — CPU-only image.
# No GPU and no Hugging Face token are required at build or run time:
#   - STT can be delegated to an OpenAI-compatible endpoint (e.g. llama-swap)
#     via OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_DEFAULT_STT_MODEL.
#   - Diarization runs locally via sherpa-onnx CPU wheels; models auto-download
#     to WHISPER_MODELS_DIR on first use (mount a volume there to persist them).
# The app (uvicorn) binds 0.0.0.0:8000 by default (see config.py HOST/PORT).
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
