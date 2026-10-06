---
type: runbook
title: "Verify the diarizer sidecar lifecycle via llama-swap"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, llama-swap, verification]
system: transcriber
see_also: ["decisions/0013-llama-swap-managed-sidecar.decision.md", "memories/0004-nemotron3-cold-start-vram-numbers.memory.md", "concepts/0002-llama-swap-generic-service-hosting.concept.md"]
---

# Runbook: diarizer sidecar lifecycle check

## Prerequisites
llama-swap running on puma with the `nemotron-3-diarization` entry; image `nemotron3-diarizer:ah11` present.

## Steps
1. Cold start: `curl -X POST http://llama-swap:8080/upstream/nemotron-3-diarization/diarize -F file=@sample.wav` — expect container start + result within ~20 s.
2. Health: `docker exec llama-swap curl http://nemotron3-diarizer:8000/health` (or via /upstream).
3. VRAM: `nvidia-smi` during inference — expect ≤ ~2.6 GiB sidecar peak.
4. Eviction: idle past `ttl: 300` → container gone (`docker ps`); next request re-cold-starts (~18 s).

## Verification
Compare against the recorded baseline in [[memories/0004-nemotron3-cold-start-vram-numbers.memory]]; large deviation = regression.

## Rollback
Remove/revert the config entry (llama-swap hot-reloads config); transcription is unaffected either way — diarization skips with a named-engine warning.
