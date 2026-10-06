---
type: concept
title: "Speaker↔segment alignment"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, alignment, transcript]
system: transcriber
see_also: ["architectures/transcriber/components/0001-transcriber-app.component.md", "concepts/0003-transcription-job-pipeline.concept.md"]
---

# Concept: Speaker↔segment alignment

## What
`align_speakers_to_segments(segments, diar_result)` (`diarization/alignment.py`) assigns speaker labels to STT segments by time-overlap (`compute_overlap`): each transcript segment takes the dominant-overlapping diarization interval's speaker. Called from both the async job path (`jobs.py:363`) and the sync path (`main.py:556`) — verified at `b7a4f03`.

## Why it matters
It is the join between the two independent analyses (STT segments vs diarization intervals). The remote engine reuses the same `DiarizationResult`/`SpeakerInterval` models, so swapping engines never touches this layer (DEC-0014 design property).

## Key Details
- Downstream consumers: transcript view (word-span rendering of speaker changes, AH-15/AH-16), SRT/VTT/ASS/JSON exports, speaker renaming in UI.
- Arrival-order relabeling happens before alignment (de-blip filter, DEC-0015) — alignment always sees clean labels.
