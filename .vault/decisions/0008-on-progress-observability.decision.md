---
type: decision
id: DEC-0008
title: "Progress & logging via one on_progress callback into the existing SSE plumbing"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [observability, ui, logging]
system: transcriber
see_also: ["specifications/0001-transcriber-service.spec.md"]
---

# DEC-0008: Progress & logging via one on_progress callback

## Context
UI silence during jobs was a pipeline-feed gap, not a frontend defect: `update_status`→SSE→app.js works; the pipeline never called it between hardcoded markers. All engines' `transcribe()` accept `**kwargs` — a callback kwarg is signature-safe.

## Decision
Adapter emits `on_progress(done_audio_s, total_audio_s, phase)` with `phase ∈ {"compressing","chunk"}`; jobs.py maps it into the 35→50 progress band and INFO logs — one feed, two consumers. Plus `LOG_LEVEL` env, per-stage duration logs, "queued"/"compressing" statuses, diarization heartbeat (no fake percentages).

## Alternatives Considered
Polling job stats (new API surface); frontend-side estimation (duplicates logic); fake time-based percentages (dishonest).

## Consequences
Zero frontend/API/schema change; tests assert status-event sequences, not log strings; upstream-able.
