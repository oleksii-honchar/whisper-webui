---
type: memory
title: "DEFAULT_WHISPER_ENGINE defaults to faster-whisper"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [config, gotcha]
system: transcriber
see_also: ["decisions/0003-config-driven-openai-default-stt-model.decision.md"]
---

# Memory: default whisper engine

**Fact:** the app's `default_whisper_engine` defaults to `faster-whisper` (config.py) — without `DEFAULT_WHISPER_ENGINE=openai` the app prefers pulling a local model instead of calling llama-swap.
**Context:** re-verified in config.py at `b7a4f03`.
**Impact:** the env var must stay set in the deployment; otherwise a local model download silently appears.
