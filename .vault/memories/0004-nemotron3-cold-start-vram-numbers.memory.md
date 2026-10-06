---
type: memory
title: "Nemotron-3 diarizer measured cold-start and VRAM numbers"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, performance, measurements]
system: transcriber
see_also: ["decisions/0013-llama-swap-managed-sidecar.decision.md", "architectures/transcriber/containers/0002-nemotron3-diarizer-sidecar.container.md"]
---

# Memory: diarizer sidecar measured numbers

**Fact (measured 2026-10-05, T9/AH-12):** model load 0.9 s; container start→health 8–9 s; cold request via `/upstream` 17 s; re-cold after ttl eviction 18 s; co-load (whisper+diarizer) 2061 MiB both-ready; inference peak 2598 MiB (activations); GPU total with RAG resident 5.7/8 GB; diarization RTF 0.0066 on a 47.6-min file.
**Context:** session evidence `materials/report-AH11-gate.md`, `materials/evidence-AH12-lifecycle.txt` — recorded measurements, not re-run.
**Impact:** cold-start semantics identical to the whisper path; acceptable for batch use; use these as the baseline for any regression check.
