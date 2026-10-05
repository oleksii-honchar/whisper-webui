# AH-11a validation harness (T9/V1) — NeMo sidecar image (spec §3.5 V1).
# NeMo deps are isolated here, NOT in the app image (DEC-12 security posture).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# nemo-toolkit 3.0.0 (PyPI, latest release) lacks self_attention_model='rope',
# which Nemotron-3-Diarization requires -> pin NeMo Speech main (verified 2026-10-05).
RUN pip install --no-cache-dir Cython packaging \
        "nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@1688cc3d6a9ade854f544987810c53f605dc86fc" \
        fastapi uvicorn
# separate layer: multipart form support (keeps the heavy NeMo layer cached)
RUN pip install --no-cache-dir python-multipart

# Bake the model into the HF cache at build time (~0.4 GB weights, OpenMDW v1.1).
# Pass --build-arg HF_TOKEN=... only if the HF download ever requires auth.
ARG HF_TOKEN=""
ENV HF_TOKEN="$HF_TOKEN"
RUN python -c "from nemo.collections.asr.models import SortformerEncLabelModel as M; M.from_pretrained('nvidia/Nemotron-3-Diarization'); print('model baked')"

WORKDIR /srv
COPY app.py .

# llama-swap cmd pattern passes PORT; default 8000 for direct runs.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
