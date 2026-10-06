---
type: index
title: "transcriber — System Context"
c4_level: system_context
system: transcriber
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [c4, transcriber]
see_also: ["specifications/0001-transcriber-service.spec.md", "decisions/0001-reuse-llama-swap-whisper-stt.decision.md", "decisions/0013-llama-swap-managed-sidecar.decision.md"]
---

# transcriber — System Context

Transcription web UI with speaker diarization, deployed on puma.lan (`https://transcriber.lan`). App: fork `oleksii-honchar/whisper-webui` of jeckyllX/transcriber.

## Diagram

```mermaid
C4Context
  title transcriber - System Context
  Person(user, "Homelab user", "Uploads audio/video for transcription")
  System(transcriber, "transcriber", "Transcription web UI with speaker diarization (fork of jeckyllX/transcriber)")
  System_Ext(caddy, "Caddy", "TLS + routing, transcriber.lan")
  System_Ext(llama_swap, "llama-swap", "Model host on puma.lan: whisper.cpp STT + diarizer sidecar, on-demand matrix")
  Rel(user, transcriber, "Uploads audio, reads transcripts", "HTTPS")
  Rel(caddy, transcriber, "reverse_proxy", "HTTP :8000")
  Rel(transcriber, llama_swap, "POST /v1/audio/transcriptions + /upstream/nemotron-3-diarization/diarize", "HTTP")
```

## Levels

- [[architectures/transcriber/containers/_index|Containers]] — app + diarizer sidecar
- [[architectures/transcriber/components/_index|Components]] — inside the app container
- Code level: not documented (class-level detail is LSP's job; key contracts in the component node)

## Related

- Specification: [[specifications/0001-transcriber-service.spec]]
- Key decisions: [[decisions/0001-reuse-llama-swap-whisper-stt.decision]], [[decisions/0013-llama-swap-managed-sidecar.decision]]
