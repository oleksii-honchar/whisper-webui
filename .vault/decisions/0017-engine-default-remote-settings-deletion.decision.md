---
type: decision
id: DEC-0017
title: "Engine default flips to remote; the five sherpa-only settings are deleted"
status: accepted
supersedes: ["DEC-0010"]
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [config, cleanup]
system: transcriber
see_also: ["decisions/0016-remove-sherpa-engine.decision.md", "decisions/0010-wire-diarization-threshold.decision.md"]
---

# DEC-0017: Engine default → remote; delete sherpa-only settings

## Context
`diarization_seg_model`, `diarization_emb_model`, `diarization_threshold`, `diarization_num_threads`, `diarization_provider` had zero consumers outside `sherpa_diarizer.py`. DEC-0010's precedent: a dead setting is a bug.

## Decision
`diarization_engine` default → `"remote"` (factory default param likewise); delete the five settings and their config.json.example keys. KEEP `diarization_enabled`, `diarization_api_url`, `diarization_request_timeout`, `diarization_min_speaker_duration`. Deployment still sets `DIARIZATION_ENGINE=remote`/`DIARIZATION_API_URL` explicitly (explicitness beats brevity for a homelab runbook).

## Alternatives Considered
Keep knobs "for future local engines" (re-creates dead settings — a future engine ships its own knobs); keep `threshold` as generic (no consumer — dead again).

## Consequences
Verified at `b7a4f03`: config.py defaults `DIARIZATION_ENGINE=remote`; only the four kept settings exist. Stale keys in a deployed config.json volume are simply not read.
