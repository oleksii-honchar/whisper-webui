---
type: concept
title: "Diarization quality evaluation without ground truth"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, evaluation, methodology]
system: transcriber
see_also: ["decisions/0012-nemotron3-gated-adoption.decision.md", "memories/0005-canonical-regression-fixture.memory.md"]
---

# Concept: Diarization quality evaluation (label-distribution standard)

## What
A ground-truth-free evaluation standard used to gate diarization mechanisms on real calls: (1) expected speaker-label count, (2) `dominant_share` ≲ 0.7 (no single speaker swallowing the file), (3) boundary spot-checks against known dialogue, (4) resource measurements (RTF, VRAM, cold-start).

## Why it matters
It made a hard architecture decision decidable: the T9 gate ran this standard on a real 47.6-min call and produced a defensible go/no-go (all clauses passed except one deterministic 0.51 s blip → DEC-0015 filter → offline re-gate RG). Deterministic model output allowed re-gating by post-processing without GPU re-runs.

## Key Details
- Judge the result JSON's label distribution, not transcripts.
- Over-clustering (91 labels on 2 speakers) and spurious-blip labels are the two failure signatures this standard catches.
- Reuse for any future engine swap: same fixture ([[memories/0005-canonical-regression-fixture.memory]]), same clauses.
