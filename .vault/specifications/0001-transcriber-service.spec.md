---
type: specification
kind: feature
title: "transcriber service — transcription web UI with diarization on puma.lan"
status: completed
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [transcriber, stt, diarization]
system: transcriber
see_also: ["architectures/transcriber/_index.md", "decisions/0001-reuse-llama-swap-whisper-stt.decision.md", "decisions/0012-nemotron3-gated-adoption.decision.md", "decisions/0016-remove-sherpa-engine.decision.md"]
---

# Specification: transcriber service

Condensed from the session spec (5 revisions). Full working document: session `specifications/spec.md` @ ses_ef93afbeeffeXCoso1uvOnC3n9.

## Goal
Self-hosted web UI for audio transcription with speaker diarization on puma.lan, reusing the llama-swap `whisper-large-v3-turbo` endpoint as STT backend. App: fork `oleksii-honchar/whisper-webui` (of jeckyllX/transcriber).

## Architecture
See [[architectures/transcriber/_index]] — app container → llama-swap (STT + `/upstream` diarizer passthrough); Nemotron-3 sidecar managed by the llama-swap matrix (`w & d`).

## Phases (all landed)
- T1 config patch (P1–P3) · T2 Dockerfile · T3 puma-lan service · T4 E2E · T5 upstream PR — ⚠️ not verified as submitted
- T6 AH-4 observability · T7 AH-5 diarization perf/quality · T8 verification (job f3dffb29 baseline → fixed)
- T9 Nemotron-3 validation gate (quality PASS except one blip; hosting resolved via AH-12) · T10 RemoteDiarizer + de-blip integration (E2E PASS) · AH-13…AH-17 follow-ups
- T11 AH-18 remove sherpa + T11-deploy (single-engine production, image `tuiteraz/whisper-webui`)

## Success criteria
1–14 all met per session evidence; criterion 14 (sherpa removal, single engine, E2E unchanged) verified against code 2026-10-06.

## Resolved risks (durable)
- Dummy API key requirement ([[memories/0001-dummy-openai-api-key-required.memory]])
- Shared openai config breaks the summarizer ([[memories/0002-shared-openai-config-breaks-summarizer.memory]])
- DEFAULT_WHISPER_ENGINE must be set to `openai` ([[memories/0003-default-whisper-engine-is-faster-whisper.memory]])
- No in-app diarization fallback since DEC-0016 — diarization skips if the sidecar path is down; rollback = revert image pin

## API contracts (durable)
- Outbound STT: `POST http://llama-swap:8080/v1/audio/transcriptions` — multipart `file` + `model=whisper-large-v3-turbo` + `response_format=verbose_json` (+ optional `language`, word timestamps with graceful 400-fallback); `verbose_json` segments map 1:1 onto the app's `Segment` model.
- Outbound diarization: `POST {DIARIZATION_API_URL}/diarize` — multipart file → `{num_speakers, intervals:[{start,end,speaker}]}` (→ `DiarizationResult`/`SpeakerInterval`); `GET /health` used only by llama-swap's checkEndpoint.
- Inbound: unchanged app API; `cluster_threshold`/`num_speakers` accepted-and-ignored by the sole engine (contract stability, DEC-0014/0018).

## Security posture (durable)
NeMo's large dependency tree is isolated in the sidecar image, not the app image; the sidecar stays on `puma-net`; audio never leaves the LAN (hosted pyannoteAPI rejected for exactly this reason); model weight licenses per [[memories/0006-model-license-constraints.memory]].

## Open decisions (still open)
Keyless-endpoints PR; llama-swap eviction tuning; fork repo rename; split STT/LLM base URLs PR; image tag policy (`:latest` vs pinned — ⚠️ current compose uses `:latest`, see Unverified); upstreaming the remote-engine pattern.
