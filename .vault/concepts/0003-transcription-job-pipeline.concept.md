---
type: concept
title: "Transcription job pipeline"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [pipeline, jobs, observability]
system: transcriber
see_also: ["architectures/transcriber/components/0001-transcriber-app.component.md", "decisions/0008-on-progress-observability.decision.md", "decisions/0011-stt-request-timeout-setting.decision.md"]
---

# Concept: Transcription job pipeline

## What
The async job flow in `jobs.py:run_pipeline`: upload → audio compression (ffmpeg → mp3) → chunking (24 MB / 20-min chunks) → per-chunk STT via the configured transcriber adapter (cumulative `time_offset` keeps honest audio-time) → diarization (remote engine, default thread executor) → speaker↔segment alignment → results/exports.

## Why it matters
All observability and reliability behavior hangs off this flow: one `on_progress(done_audio_s, total_audio_s, phase)` callback feeds both INFO logs and the UI progress band (queued=5, transcribing 35→50, …); a semaphore serializes jobs; a diarization heartbeat covers the remote cold-start window; failure paths log with the engine named and diarization skips (warning) without blocking transcription.

## Key Details
- Chunk loop is resumable per-chunk: per-chunk INFO logs (i/N, offset, latency) make partial progress diagnosable.
- STT request timeout is a setting (default 600 s) — sized for 20-min chunks behind a cold-start llama-swap.
- Sync `/transcribe` path (main.py) mirrors the same flow without the job queue.
