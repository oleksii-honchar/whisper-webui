---
type: decision
id: DEC-0013
title: "Hosting: llama-swap-managed sidecar (matrix var d, puma_asr: w & d)"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [hosting, llama-swap, gpu]
system: transcriber
see_also: ["concepts/0002-llama-swap-generic-service-hosting.concept.md", "architectures/transcriber/containers/0002-nemotron3-diarizer-sidecar.container.md", "runbooks/0002-diarizer-sidecar-lifecycle-check.runbook.md"]
---

# DEC-0013: llama-swap-managed diarizer sidecar

## Context
User's "exclusive mode with whisper" instinct; config already runs the matrix DSL (vars/evict_costs/sets, exclusive `qw9b` precedent); llama-swap serves any OpenAI-compatible server via `cmd` and exposes `/upstream/{model_id}/…` passthrough; external containers are invisible to its VRAM solver.

## Decision
Register the diarizer as a llama-swap model (`cmd: docker run --gpus all …` + `cmdStop`, `checkEndpoint: /health`, `ttl: 300`), matrix var `d`, `puma_asr: "w & d"`. Fork calls `http://llama-swap:8080/upstream/nemotron-3-diarization/diarize`.

## Alternatives Considered
Plain always-on sidecar (solver-invisible co-residency gamble); NeMo in-process in the fork (heavy deps in app image); hosted pyannoteAPI (audio leaves the LAN).

## Consequences
VRAM arbitration explicit and shared with whisper's ttl/eviction machinery. Landed via AH-12 (llama-swap image extended with docker CLI + socket mount); verified live in config-1.yaml at puma-lan.
