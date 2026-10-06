---
type: decision
id: DEC-0009
title: "Diarization performance: multi-thread CPU + int8 + process-pool; GPU NOT adopted"
status: superseded
superseded_by: ["DEC-0016", "DEC-0018"]
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, performance, gpu]
system: transcriber
see_also: ["decisions/0016-remove-sherpa-engine.decision.md", "concepts/0001-diarization-clustering-length-degradation.concept.md"]
---

# DEC-0009: Diarization performance — CPU multi-thread first, GPU rejected

> Superseded 2026-10-06 by DEC-0016/DEC-0018 (the sherpa engine and the process-pool machinery were removed). The GPU-rejection reasoning remains valid background for any future in-app engine.

## Context
The fork never set `num_threads` → single-threaded CPU; observed RTF 0.33 matched the official 1-thread fp32 figure 0.297. GPU evidence (k2-fsa discussions #2054/#3233): CUDA RTF ≈0.24 ≈ single-thread CPU; segmentation model poorly GPU-utilized; CUDA wheel +250 MB; VRAM contention; GIL freeze unchanged by GPU.

## Decision
`num_threads` config-driven (default min(8, cpu_count)); int8 segmentation via existing setting (deployment env only); `ProcessPoolExecutor(max_workers=1)` to defeat the GIL freeze. GPU stayed a config experiment, never shipped.

## Alternatives Considered
CUDA wheel by default (marginal gain, +250 MB, VRAM contention); NeMo titanet_small embedding (EN-trained, parked).

## Consequences
Large speedup on the 12-core host; the pool machinery was later deleted by DEC-0018.
