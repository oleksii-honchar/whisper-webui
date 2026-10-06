---
type: decision
id: DEC-0010
title: "Wire the dead diarization_threshold setting; raise default to 0.75"
status: superseded
superseded_by: ["DEC-0017"]
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, quality, config]
system: transcriber
see_also: ["decisions/0017-engine-default-remote-settings-deletion.decision.md"]
---

# DEC-0010: Wire the dead diarization_threshold setting

> Superseded by DEC-0017 (setting deleted with the sherpa engine). Its precedent — **a dead setting is a bug, not a spare knob** — remains an active principle in this codebase.

## Context
91 speakers on a 2-person 47-min call. `config.py` defined `diarization_threshold` but three hardcoded 0.5 defaults (API Form, run_pipeline, diarize()) made it dead code.

## Decision
Wire all three defaults to the setting and raise 0.5 → 0.75; validate on a labeled sample.

## Alternatives Considered
0.7/0.8 (unvalidated); UI-exposed tuning (scope creep); fixed num_speakers hint (wrong for unknown-count files).

## Consequences
One real bug fixed; measurable quality target; per-deployment override documented.
