---
type: decision
id: DEC-0006
title: "CPU-only runtime; models on a persistent volume"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [deployment, resources]
system: transcriber
see_also: ["decisions/0001-reuse-llama-swap-whisper-stt.decision.md", "decisions/0016-remove-sherpa-engine.decision.md"]
---

# DEC-0006: CPU-only runtime; models on a persistent volume

## Context
puma GPU is contended (llama-swap RAG set ~7.0–7.5 GB of 8 GB when loaded); in-app diarization was CPU ONNX with auto-download.

## Decision
No GPU allocation, no HF token for the app container; mount a persistent volume for `WHISPER_MODELS_DIR` so models survive recreates.

## Alternatives Considered
GPU int8 faster-whisper in-container (rejected — DEC-0001); re-download on each recreate (wasteful).

## Consequences
Predictable resource profile. Update: after DEC-0016 the models volume serves STT-only (diarization is remote).
