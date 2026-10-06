---
type: decision
id: DEC-0016
title: "Remove the sherpa-onnx diarization engine entirely"
status: accepted
supersedes: []
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, cleanup]
system: transcriber
see_also: ["decisions/0014-remote-diarizer-engine-switch.decision.md", "decisions/0017-engine-default-remote-settings-deletion.decision.md", "decisions/0018-remove-process-pool-machinery.decision.md"]
---

# DEC-0016: Remove the sherpa-onnx engine (supersedes DEC-0014's rollback role)

## Context
DEC-14 kept sherpa as default + one-env-var rollback. That rationale expired: the remote path passed its E2E gate and is the live engine; sherpa's FastClustering length-degradation is the very defect the remote engine replaced; keeping it means a runtime dependency, five config knobs, ~15 tests, and doc surface nobody uses. User direction explicit ("remove sherpa").

## Decision
Delete `diarization/sherpa_diarizer.py`, its registration/exports, and all code that exists only for it. The factory registry and `DIARIZATION_ENGINE` switch STAY — the extension surface the remote engine relies on. Rollback changes character: revert the compose image pin (the image stays published), not flip an env var.

## Alternatives Considered
Keep sherpa disabled (still imports the dependency, still maintained); keep file without registration (dead code); hardcode the remote call (throws away the plugin surface).

## Consequences
Verified complete at `b7a4f03`: `rg -i sherpa` → 0 matches in the fork. Loss of in-app diarization fallback recorded: if the sidecar path is down, diarization skips with a named-engine warning; transcription unaffected.
