---
type: decision
id: DEC-0005
title: "Dummy non-empty OPENAI_API_KEY"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [config, deployment]
system: transcriber
see_also: ["memories/0001-dummy-openai-api-key-required.memory.md", "memories/0002-shared-openai-config-breaks-summarizer.memory.md"]
---

# DEC-0005: Dummy non-empty OPENAI_API_KEY

## Context
`OpenAICompatibleTranscriber.is_available()` returns `bool(api_key)` — without a key the engine is invisible; llama-swap ignores any key.

## Decision
Set a non-empty dummy value in Infisical `/transcriber`.

## Alternatives Considered
Patch `is_available()` to allow keyless custom endpoints (extra fork diff; optional future PR).

## Consequences
Zero code change; must stay documented. Side effect: `openai_base_url/api_key` are shared with the LLM summarization provider — the OpenAI-compat summarizer is unusable in this deployment (accepted for v1; split-URL PR is an open decision).
