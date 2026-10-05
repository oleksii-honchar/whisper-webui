# AH-11a validation harness (T9/V1) — Nemotron-3-Diarization sidecar.
# FastAPI wrapper around NeMo SortformerEncLabelModel for the T9 gate ONLY.
# Not integrated into the app (T10 decides that). Spec §3.5 V1, DEC-12/13.
#
# Endpoints:
#   GET  /health   -> {status, ready, model, load_seconds, cuda}
#   POST /diarize  -> multipart "file" -> {num_speakers, intervals:[{start,end,speaker}]}
#
# Build: see Dockerfile (model baked into the HF cache at build time).
# Runtime env: PORT (default 8000), DIAR_MODEL_ID (default nvidia/Nemotron-3-Diarization),
#              HF_TOKEN (only if the HF download requires it).

import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from nemo.collections.asr.models import SortformerEncLabelModel

MODEL_ID = os.getenv("DIAR_MODEL_ID", "nvidia/Nemotron-3-Diarization")
# Model-card offline configuration (30.4 s latency), units = 80 ms frames.
STREAM_CFG = {"spkcache_len": 264, "fifo_len": 40, "chunk_len": 340,
              "chunk_right_context": 40, "spkcache_update_period": 300}
NATIVE_FMTS = {".wav", ".flac", ".opus", ".mp3"}

app = FastAPI(title="nemotron-3-diarizer")
_model = None
_load_seconds = None


@app.on_event("startup")
def _load_model():
    global _model, _load_seconds
    t0 = time.monotonic()
    m = SortformerEncLabelModel.from_pretrained(MODEL_ID)
    m.eval()
    for k, v in STREAM_CFG.items():
        setattr(m.sortformer_modules, k, v)
    m._check_streaming_parameters()
    if torch.cuda.is_available():
        m.to("cuda")
    _model = m
    _load_seconds = time.monotonic() - t0
    print(f"model loaded in {_load_seconds:.1f}s device={m.device}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading",
            "ready": _model is not None, "model": MODEL_ID,
            "load_seconds": _load_seconds, "cuda": torch.cuda.is_available()}


def _to_native(src: Path) -> Path:
    """NeMo accepts wav/flac/opus/mp3; convert anything else via ffmpeg (16k mono)."""
    if src.suffix.lower() in NATIVE_FMTS:
        return src
    dst = src.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1",
                    "-vn", str(dst)], check=True, capture_output=True)
    return dst


def _seg_fields(seg):
    if isinstance(seg, str):  # NeMo diarize() output: "start end speaker_N"
        start, end, speaker = seg.split()
        return float(start), float(end), speaker
    if hasattr(seg, "start_time"):  # namedtuple fallback
        return float(seg.start_time), float(seg.end_time), str(seg.speaker)
    start, end, speaker = seg  # tuple fallback: begin, end, speaker_index
    return float(start), float(end), str(speaker)


@app.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="model loading")
    suffix = Path(file.filename or "audio").suffix or ".wav"
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("in" + suffix)
        src.write_bytes(await file.read())
        audio = _to_native(src)
        t0 = time.monotonic()
        predicted = _model.diarize(audio=[str(audio)], batch_size=1)
        wall = time.monotonic() - t0
    labels, intervals = {}, []
    for seg in predicted[0]:
        start, end, speaker = _seg_fields(seg)
        if speaker not in labels:
            labels[speaker] = f"spk{len(labels)}"  # arrival-order stable labels
        intervals.append({"start": round(start, 3), "end": round(end, 3),
                          "speaker": labels[speaker]})
    intervals.sort(key=lambda i: i["start"])
    print(f"diarize wall={wall:.1f}s intervals={len(intervals)} "
          f"speakers={len(labels)}", flush=True)
    return {"num_speakers": len(labels), "intervals": intervals}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
