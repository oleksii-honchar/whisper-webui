---
type: index
title: "Decisions"
createdAt: "2026-10-06T13:19:26Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: []
---

# Decisions

Architecture decision records for whisper-webui / transcriber, grouped by theme. Filename NNNN = source of truth for `DEC-NNNN`.

## STT backend & config

- [[0001-reuse-llama-swap-whisper-stt.decision]] — llama-swap as STT backend (accepted)
- [[0003-config-driven-openai-default-stt-model.decision]] — config-driven model profile patch (accepted)
- [[0005-dummy-openai-api-key.decision]] — dummy key requirement (accepted)
- [[0011-stt-request-timeout-setting.decision]] — 600 s STT timeout setting (accepted)

## Fork & branch

- [[0002-service-name-transcriber.decision]] — naming (accepted)
- [[0004-patch-branch-and-upstream-pr.decision]] — branch + upstream PR strategy (accepted)

## Deployment

- [[0006-cpu-only-runtime-models-volume.decision]] — CPU-only runtime, models volume (accepted)
- [[0007-dockerfile-for-fork.decision]] — Dockerfile (accepted)

## Observability

- [[0008-on-progress-observability.decision]] — on_progress callback → logs + SSE (accepted)

## Diarization engine evolution

- [[0009-cpu-diarization-performance-no-gpu.decision]] — CPU perf path, GPU rejected (**superseded** by 0016/0018; GPU-rejection reasoning still valid)
- [[0010-wire-diarization-threshold.decision]] — threshold wiring (**superseded** by 0017; "dead setting is a bug" precedent stands)
- [[0012-nemotron3-gated-adoption.decision]] — Nemotron-3, gated adoption (accepted)
- [[0013-llama-swap-managed-sidecar.decision]] — sidecar hosting (accepted)
- [[0014-remote-diarizer-engine-switch.decision]] — RemoteDiarizer + engine switch (accepted; rollback role superseded by 0016)
- [[0015-de-blip-min-speaker-duration.decision]] — de-blip filter (accepted)
- [[0016-remove-sherpa-engine.decision]] — sherpa removal (accepted)
- [[0017-engine-default-remote-settings-deletion.decision]] — default flip + settings deletion (accepted)
- [[0018-remove-process-pool-machinery.decision]] — process-pool removal (accepted)
