---
type: decision
id: DEC-0001
title: "Reuse the existing llama-swap Whisper endpoint as STT backend"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [stt, llama-swap, deployment]
system: transcriber
see_also: ["specifications/0001-transcriber-service.spec.md", "architectures/transcriber/containers/0001-transcriber.container.md"]
---

# DEC-0001: Reuse the existing llama-swap Whisper endpoint as STT backend

## Context
User wanted a diarization-capable transcription web UI; the homelab already serves `whisper-large-v3-turbo` via llama-swap (config-1.yaml), live-probed with OpenAI-style `verbose_json`.

## Decision
The app calls `http://llama-swap:8080/v1/audio/transcriptions` container-to-container on `puma-net`.

## Alternatives Considered
- Local faster-whisper in-container — duplicates a model, VRAM co-residency gamble on the 8 GB card.
- Open WebUI STT — no diarization/file workspace.
- Hosted HF Space — not self-hosted.

## Consequences
Single shared model instance; inherits cold-start latency (ttl 300, evict-first) — accepted for batch use.
