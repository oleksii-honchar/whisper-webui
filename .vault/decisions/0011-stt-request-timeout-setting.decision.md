---
type: decision
id: DEC-0011
title: "STT request timeout becomes a setting (default 600 s)"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [config, reliability]
system: transcriber
see_also: ["decisions/0001-reuse-llama-swap-whisper-stt.decision.md"]
---

# DEC-0011: STT request timeout setting

## Context
`httpx.Timeout(180.0)` — a 20-min chunk behind a cold-start llama-swap (≈15–20 s model load + transcription) can exceed 180 s → silent timeout failure.

## Decision
`stt_request_timeout` setting (env `STT_REQUEST_TIMEOUT`, default 600), used by `_call_transcription_api`.

## Alternatives Considered
Hardcode 600 (unflexible); retry logic (masks the issue, doubles load on a cold model).

## Consequences
Cold-start + long-chunk requests survive; with per-chunk logs, timeouts become diagnosable.
