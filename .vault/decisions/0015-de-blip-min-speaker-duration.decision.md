---
type: decision
id: DEC-0015
title: "De-blip policy: per-speaker min-total-duration filter inside RemoteDiarizer"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, quality, post-processing]
system: transcriber
see_also: ["decisions/0012-nemotron3-gated-adoption.decision.md", "memories/0004-nemotron3-cold-start-vram-numbers.memory.md"]
---

# DEC-0015: De-blip policy (min speaker duration filter)

## Context
Nemotron-3 is end-to-end; its residual error class on long audio is spurious speaker attribution — one 0.51 s fragment labeled spk2 out of 885 intervals (deterministic across runs). Real speakers: 564.7 s / 1843.6 s — four orders of magnitude above the artifact. Upstream has no filtering mechanism.

## Decision
Inside `RemoteDiarizer.diarize()`: drop every speaker whose TOTAL speech duration < `settings.diarization_min_speaker_duration` (env `DIARIZATION_MIN_SPEAKER_DURATION`, default 2.0 s; 0 disables), re-label survivors in arrival order, recompute `num_speakers`, INFO-log each drop.

## Alternatives Considered
Per-interval filter (chops legitimate short turns); filter in the sidecar (buries raw model behavior); `num_speakers=2` hint (breaks unknown-count files); accept 3 labels (pushes the heuristic into every consumer).

## Consequences
Verified in config.py at `b7a4f03`. Known trade-off: a genuine brief speaker on a short file could be erased — mitigated by the knob, the 2.0 s near-noise default, and INFO logs naming every drop.
