---
type: decision
id: DEC-0004
title: "Branch strategy: patched/main + upstream PR"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [git, upstream, fork]
system: transcriber
see_also: ["decisions/0003-config-driven-openai-default-stt-model.decision.md"]
---

# DEC-0004: Branch strategy: patched/main + upstream PR

## Context
Prepared branch `pacthed/main` (typo) existed, identical to `main` @ `e60f496`; upstream is active.

## Decision
Rename branch to `patched/main`, keep patches as a small branch, submit PR upstream; build images from pinned patched commits.

## Alternatives Considered
Silent permanent fork (rebase burden grows); PR-only without branch (build timing risk).

## Consequences
Drift risk bounded. Update in practice: the improvement chain continued on `feat/261004-1952-improvements` (current working branch); the strategy — small branch, upstream-able patches — stands. ⚠️ Upstream PR submission not verified as of 2026-10-06.
