---
type: memory
title: "Canonical diarization regression fixture: 47.6-min Ukrainian two-person call"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [testing, diarization, fixture]
system: transcriber
see_also: ["concepts/0005-diarization-quality-eval-standard.concept.md", "memories/0004-nemotron3-cold-start-vram-numbers.memory.md"]
---

# Memory: canonical regression fixture

**Fact:** the 47.6-min Ukrainian conversation call is this project's canonical diarization test: expected result = exactly 2 speakers, dominant_share 0.645, 4/4 boundary spot-checks; real speakers' totals 564.7 s / 1843.6 s. Raw model outputs are saved (session `data-ah11-full-result.json` raw 3-label output, `data-T10E2E-full-result.json` post-filter output) for offline re-gating.
**Context:** used by the T9 gate, the RG re-gate, and the T10 E2E; model output verified deterministic across runs.
**Impact:** any future diarization engine change or filter change must be re-gated against this fixture before landing.
