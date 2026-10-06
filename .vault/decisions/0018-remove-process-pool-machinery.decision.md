---
type: decision
id: DEC-0018
title: "Delete the process-pool machinery and use_process_pool; threshold defaults become the contract constant"
status: accepted
supersedes: []
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [cleanup, concurrency]
system: transcriber
see_also: ["decisions/0009-cpu-diarization-performance-no-gpu.decision.md", "decisions/0016-remove-sherpa-engine.decision.md"]
---

# DEC-0018: Remove process-pool machinery

## Context
`_diarize_worker`, the `ProcessPoolExecutor` branch, the `use_process_pool` flag, and the `models_ready`/`ensure_models` hasattr block existed for one reason: sherpa's blocking, GIL-holding `process()`. With one HTTP-backed engine the thread-executor path is the only path.

## Decision
Remove the pool machinery and the flag; diarization always runs in the default thread executor. `cluster_threshold`/`num_speakers` remain public API + `BaseDiarizer` contract parameters (accepted-and-ignored), defaults now the contract constant `0.5`. The diarization heartbeat STAYS (covers the ~17 s remote cold-start window).

## Alternatives Considered
Keep the pool branch "just in case" (dead code + pickling machinery); remove the API params (breaking change for zero benefit); remove the heartbeat (regresses observability).

## Consequences
Re-adding any local engine reintroduces the pattern from git history (documented in DEC-0009/DEC-0014).
