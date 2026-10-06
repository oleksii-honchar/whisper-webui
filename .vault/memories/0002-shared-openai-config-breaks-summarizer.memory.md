---
type: memory
title: "openai_base_url/api_key are shared between STT and the LLM summarizer"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [config, gotcha]
system: transcriber
see_also: ["decisions/0005-dummy-openai-api-key.decision.md"]
---

# Memory: shared openai_compatible config

**Fact:** `llm/registry.py` reads the same `openai_base_url/api_key/default_model` as the STT profile — pointing the base URL at llama-swap makes the OpenAI-compat **summarizer unusable** (chat request to a whisper endpoint).
**Context:** accepted for v1; UI summarization must use Ollama or stay disabled.
**Impact:** clean fix = split STT/LLM base URLs (upstream PR candidate, open decision).
