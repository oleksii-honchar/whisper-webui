---
type: container
c4_level: container
title: "transcriber app container"
system: transcriber
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [c4, container, transcriber]
system: transcriber
see_also: ["decisions/0007-dockerfile-for-fork.decision.md", "runbooks/0001-transcriber-image-chain-deploy.runbook.md", "architectures/transcriber/components/0001-transcriber-app.component.md"]
linked_elements: ["transcriber_app", "llama_swap", "caddy"]
---

# Container: transcriber

## Diagram

```mermaid
C4Container
  title transcriber system - containers on puma.lan (puma-net)
  Person(user, "Homelab user", "")
  System_Ext(caddy, "Caddy", "TLS + routing")
  System_Ext(llama_swap, "llama-swap :8080", "whisper.cpp STT + hosts the diarizer via cmd/cmdStop, matrix w & d")
  Container(transcriber_app, "transcriber", "Python 3.12-slim, FastAPI, ffmpeg", "Transcription + diarization web app, port 8000")
  Container(diarizer_sidecar, "nemotron3-diarizer", "NeMo SortformerEncLabelModel + FastAPI", "End-to-end diarization: POST /diarize, GET /health; GPU ~0.4 GB, on-demand")
  Rel(user, transcriber_app, "Uploads audio, reads transcripts", "HTTPS via transcriber.lan")
  Rel(caddy, transcriber_app, "reverse_proxy transcriber:8000")
  Rel(transcriber_app, llama_swap, "STT + diarization via /upstream passthrough", "HTTP")
  Rel(llama_swap, diarizer_sidecar, "docker run/stop, checkEndpoint /health, ttl 300", "docker.sock")
```

## Elements

| ID | Name | Type | Technology | Description |
|----|------|------|-----------|-------------|
| transcriber_app | transcriber | Container | Python 3.12-slim, FastAPI, ffmpeg, vanilla JS | Web UI + job pipeline; image `tuiteraz/whisper-webui` built from the fork |
| llama_swap | llama-swap | System_Ext | llama-swap + whisper.cpp | STT backend + diarizer host on `puma-net` |
| diarizer_sidecar | nemotron3-diarizer | Container | NeMo, FastAPI | See [[architectures/transcriber/containers/0002-nemotron3-diarizer-sidecar.container]] |

## Notes

- Volumes: `transcriber-models` (STT-only since DEC-0016), `transcriber-config`, outputs.
- Env via Infisical `/transcriber`; dummy `OPENAI_API_KEY` required ([[memories/0001-dummy-openai-api-key-required.memory]]).
- ⚠️ compose currently references `tuiteraz/whisper-webui:latest` — spec intent was pinned commit (open decision).
