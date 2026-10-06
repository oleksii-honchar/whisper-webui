---
type: decision
id: DEC-0014
title: "Integration: RemoteDiarizer + DIARIZATION_ENGINE switch"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, integration, plugin-surface]
system: transcriber
see_also: ["architectures/transcriber/components/0001-transcriber-app.component.md", "decisions/0016-remove-sherpa-engine.decision.md"]
---

# DEC-0014: RemoteDiarizer + engine switch

> Note: the "sherpa stays default and rollback" role was superseded by DEC-0016; the RemoteDiarizer + factory/switch design itself is the live production architecture.

## Context
Verified plugin surface — `DiarizerFactory.register()` + `BaseDiarizer` contract; engine hardcoded at exactly 3 call sites; `openai_compat.py` is the codebase's own config-driven HTTP-client pattern.

## Decision
New `diarization/remote_diarizer.py` (multipart POST → `DiarizationResult`; `is_available()` = config presence, no health probe — a probe through `/upstream` would trigger model loads); settings `diarization_engine`, `diarization_api_url`, `diarization_request_timeout`; all call sites read the setting. `num_speakers`/`cluster_threshold` accepted-and-ignored by the remote engine.

## Alternatives Considered
Replace sherpa entirely at that time (no zero-risk rollback — later reversed by DEC-0016); volume-path audio sharing (couples filesystems); health probe in `is_available()` (triggers cold starts).

## Consequences
Alignment/UI/exports untouched; engine listing shows all engines for free; upstream-able as a generic remote-diarizer companion.
