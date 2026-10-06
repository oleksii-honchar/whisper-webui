---
type: concept
title: "llama-swap as a generic OpenAI-compatible service host"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [llama-swap, hosting, homelab]
system: transcriber
see_also: ["decisions/0013-llama-swap-managed-sidecar.decision.md", "runbooks/0002-diarizer-sidecar-lifecycle-check.runbook.md"]
---

# Concept: llama-swap as a generic service host

## What
llama-swap serves **any** OpenAI/Anthropic-compatible server, not just llama.cpp: model entries are `cmd:`/`cmdStop:` commands; requests route through `/upstream/{model_id}/…` passthrough (any path, e.g. `/diarize`).

## Why it matters
Registering a GPU service in llama-swap makes it **visible to the VRAM solver** — ttl eviction, `evict_costs`, and matrix sets (`vars`, `puma_asr: "w & d"`) arbitrate GPU co-residency explicitly. External containers are invisible to the solver (the co-residency gamble).

## Key Details
- First container-managed entry in this homelab's config landed via AH-12: llama-swap image extended with docker CLI + `docker.sock` mount.
- `checkEndpoint` gates readiness; a call through `/upstream` triggers the cold start (~17 s measured for the diarizer).
- Any `GET /health`-exposing container fits: `cmd: docker run --rm --network puma-net --gpus all <image>`, `cmdStop: docker stop <name>`, `ttl: 300`.
