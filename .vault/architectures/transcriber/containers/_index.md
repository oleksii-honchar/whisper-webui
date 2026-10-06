---
type: index
title: "transcriber — Containers"
c4_level: container
system: transcriber
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [c4, transcriber]
see_also: ["architectures/transcriber/_index.md"]
---

# Containers

Deployable units of the transcriber system on puma.lan (`puma-net`).

## Nodes

- [[architectures/transcriber/containers/0001-transcriber.container|transcriber app]] — FastAPI app + Web UI, image `tuiteraz/whisper-webui`
- [[architectures/transcriber/containers/0002-nemotron3-diarizer-sidecar.container|nemotron3-diarizer sidecar]] — GPU diarization, built from `diarizer-service/`

llama-swap and Caddy are homelab infrastructure (System_Ext in the diagrams), not nodes of this system.
