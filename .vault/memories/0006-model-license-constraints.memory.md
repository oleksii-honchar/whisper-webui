---
type: memory
title: "Model weight licenses constrain engine choices"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [licensing, models]
system: transcriber
see_also: ["decisions/0012-nemotron3-gated-adoption.decision.md"]
---

# Memory: model license posture

**Fact:** Nemotron-3-Diarization weights = OpenMDW v1.1 (commercial OK); pyannote Community-1 = CC-BY-4.0 + gated HF token (free, account step); DiariZen weights = CC-BY-NC — rejected for that reason; NeMo toolkit = Apache-2.0; llama-swap = MIT.
**Context:** license screen was part of the DEC-0012 candidate filter.
**Impact:** evaluate weight licenses before benchmark effort — a non-commercial clause disqualifies regardless of DER.
