---
type: decision
id: DEC-0003
title: "Patch shape: config-driven openai_default_stt_model, mirroring the codebase's own pattern"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [patch, config, upstream]
system: transcriber
see_also: ["decisions/0004-patch-branch-and-upstream-pr.decision.md", "memories/0003-default-whisper-engine-is-faster-whisper.memory.md"]
---

# DEC-0003: Patch shape: config-driven openai_default_stt_model

## Context
The UI model dropdown was hardcoded to `whisper-1` for the `openai` profile (factory.py); groq/openrouter profiles are already config-driven. Code archaeology: the hardcode came from commit `ed92151` whose stated intent is vendor-agnostic profiles — an oversight.

## Decision
Add `openai_default_stt_model` (config.json `openai_compatible.default_stt_model` / env `OPENAI_DEFAULT_STT_MODEL`, default `whisper-1`) + factory wiring + tests. No change to the adapter, UI, or settings API.

## Alternatives Considered
Hardcode `whisper-large-v3-turbo` (un-upstream-able); fork-wide rewrite (overkill).

## Consequences
2-file diff; upstream-able PR; settings UI gains the field for free via generic attr mapping. Verified present in config.py (line ~134) at `b7a4f03`.
