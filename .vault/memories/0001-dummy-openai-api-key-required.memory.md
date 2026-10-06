---
type: memory
title: "OpenAI-compat engine requires a non-empty API key"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [config, gotcha]
system: transcriber
see_also: ["decisions/0005-dummy-openai-api-key.decision.md"]
---

# Memory: dummy OPENAI_API_KEY required

**Fact:** `OpenAICompatibleTranscriber.is_available()` returns `bool(api_key)` — with an empty key the engine silently disappears from the UI/API. llama-swap ignores any key.
**Context:** verified in `transcribers/openai_compat.py`; deployment sets a dummy value in Infisical `/transcriber`.
**Impact:** never remove the dummy key; a "why is there a fake key" question is answered by DEC-0005.
