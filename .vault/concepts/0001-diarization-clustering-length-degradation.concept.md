---
type: concept
title: "Diarization clustering length-degradation"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, root-cause]
system: transcriber
see_also: ["decisions/0012-nemotron3-gated-adoption.decision.md", "decisions/0016-remove-sherpa-engine.decision.md"]
---

# Concept: Diarization clustering length-degradation

## What
Sequential/global clustering over per-segment embeddings degrades as audio length grows: thousands of embeddings → over-clustering. Observed: 91 speakers on a 2-person 47-min call (sherpa FastClustering, upstream defect k2-fsa #1466).

## Why it matters
It is the root cause that drove this project's diarization architecture. The failure lives in the **clustering layer** — not segmentation, quantization, embeddings, or display. Threshold/num_clusters tuning cannot fix it (levers exhausted across AH-9/10/10b/10c).

## Key Details
- Mechanisms that avoid it: (a) end-to-end models with no clustering (Nemotron-3 Sortformer line — chosen), (b) length-robust clustering (pyannote VBx + resegmentation — fallback), (c) windowing to validated short lengths + global speaker linking (parked).
- Residual error class of end-to-end models on long audio: spurious short speaker attributions ("blips") — handled by a per-speaker min-total-duration filter (DEC-0015), not by per-interval filtering.
