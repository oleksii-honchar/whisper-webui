---
type: decision
id: DEC-0012
title: "Diarization mechanism: NVIDIA Nemotron-3-Diarization, adoption GATED by validation"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, model-selection]
system: transcriber
see_also: ["concepts/0001-diarization-clustering-length-degradation.concept.md", "decisions/0013-llama-swap-managed-sidecar.decision.md", "decisions/0014-remote-diarizer-engine-switch.decision.md"]
---

# DEC-0012: Nemotron-3-Diarization, gated adoption

## Context
The proven failure is the clustering layer (sequential FastClustering over thousands of embeddings, k2-fsa #1466); levers exhausted. Candidates filtered to: no-clustering (M1 Nemotron-3 end-to-end), length-robust clustering (M2 pyannote Community-1 VBx), windowing (M4). M1: #1 VoiceArena 14.72% DER, 100M params ≈0.4 GB, no max duration, OpenMDW v1.1.

## Decision
Pursue M1, hard-gated by validation on the real 47.6-min Ukrainian call (label-distribution standard). Fallback if the gate fails: M2. M4 parked; M3 (DiariZen) rejected on CC-BY-NC weights.

## Alternatives Considered
Adopt-and-see (wasted integration on the one unproven axis); M2 first (token + slower — kept as fallback); keep tuning sherpa (contradicted by evidence).

## Consequences
Gate ran (T9): all clauses passed except one deterministic 0.51 s blip → resolved via DEC-0015; Ukrainian quality CONFIRMED (dominant_share 0.645, boundaries 4/4).
