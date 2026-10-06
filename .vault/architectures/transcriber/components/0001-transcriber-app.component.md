---
type: component
c4_level: component
title: "transcriber app — components"
system: transcriber
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [c4, components, transcriber]
system: transcriber
see_also: ["architectures/transcriber/containers/0001-transcriber.container.md", "decisions/0014-remote-diarizer-engine-switch.decision.md", "decisions/0008-on-progress-observability.decision.md"]
linked_elements: ["ui", "routes", "jobs", "transcribers", "diarization", "config"]
---

# Components inside the transcriber app container

## Diagram

```mermaid
C4Component
  title transcriber app container - components
  Container_Boundary(app, "transcriber container") {
    Component(ui, "Web UI", "templates/ + static/app.js", "Upload, SSE progress, transcript + speaker view")
    Component(routes, "API routes", "FastAPI main.py", "REST + SSE endpoints, settings API")
    Component(jobs, "Job pipeline", "jobs.py", "Async job orchestration, semaphore, progress band, logs")
    Component(transcribers, "Transcriber adapters", "transcribers/", "OpenAI-compatible STT client, compression, chunking, on_progress")
    Component(diarization, "Diarization plugin surface", "diarization/", "DiarizerFactory, BaseDiarizer, RemoteDiarizer (de-blip filter), alignment")
    Component(config, "Settings", "config.py", "config.json + env, settings API mapping")
  }
  System_Ext(llama_swap, "llama-swap", "STT + diarizer host")
  Rel(ui, routes, "fetch + SSE", "HTTP")
  Rel(routes, jobs, "submits jobs")
  Rel(jobs, transcribers, "transcribe(on_progress)")
  Rel(jobs, diarization, "diarize()")
  Rel(transcribers, llama_swap, "POST /v1/audio/transcriptions", "HTTP")
  Rel(diarization, llama_swap, "POST /upstream/nemotron-3-diarization/diarize", "HTTP")
  Rel(config, jobs, "engine/timeout/threshold settings")
```

## Key contracts (code-level anchors)

- `BaseDiarizer` (`diarization/base.py`): `is_available()`, `diarize(audio, num_speakers, cluster_threshold) -> DiarizationResult{num_speakers, intervals[]}`.
- `DiarizerFactory` (`diarization/factory.py`): registry; registers `RemoteDiarizer` only; `get_diarizer(name="remote")`; `list_engines()` feeds `/api/status`.
- `RemoteDiarizer` (`diarization/remote_diarizer.py`): HTTP client + DEC-0015 de-blip filter.
- `align_speakers_to_segments` (`diarization/alignment.py`): maps diarization onto STT segments.

All verified against `diarization/` at `b7a4f03`.
