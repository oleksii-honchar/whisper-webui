---
type: container
c4_level: container
title: "nemotron3-diarizer sidecar"
system: transcriber
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [c4, container, diarization, gpu]
system: transcriber
see_also: ["decisions/0012-nemotron3-gated-adoption.decision.md", "decisions/0013-llama-swap-managed-sidecar.decision.md", "concepts/0002-llama-swap-generic-service-hosting.concept.md", "runbooks/0002-diarizer-sidecar-lifecycle-check.runbook.md"]
linked_elements: ["diarizer_sidecar", "llama_swap"]
---

# Container: nemotron3-diarizer sidecar

Built from `diarizer-service/` in the fork repo (FastAPI wrapper ~100 LOC around `SortformerEncLabelModel.diarize`). Image `nemotron3-diarizer:ah11`, model baked at build (~0.4 GB).

## API

- `POST /diarize` — multipart audio → `{num_speakers, intervals:[{start,end,speaker}]}`
- `GET /health` — llama-swap `checkEndpoint` only (the app never probes)

## Lifecycle

Managed by llama-swap: `cmd: docker run --rm --name nemotron3-diarizer --network puma-net --gpus all nemotron3-diarizer:ah11`, `cmdStop: docker stop …`, `ttl: 300`, matrix var `d`, `puma_asr: "w & d"`. Measured: load 0.9 s, cold request via `/upstream` 17 s, co-load 2061 MiB both-ready, inference peak 2598 MiB — see [[memories/0004-nemotron3-cold-start-vram-numbers.memory]].

## Notes

Raw model output; the de-blip policy lives app-side (DEC-0015). Verified live in `puma-lan/llama-swap/configs/config-1.yaml` (entry `nemotron-3-diarization`, lines ~346-356).
