---
type: decision
id: DEC-0007
title: "Build a Dockerfile (python:3.12-slim + ffmpeg + uvicorn, port 8000)"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [container, deployment]
system: transcriber
see_also: ["runbooks/0001-transcriber-image-chain-deploy.runbook.md"]
---

# DEC-0007: Build a Dockerfile

## Context
No Dockerfile upstream (verified: only install.sh/run.sh); app entry `uvicorn.run("main:app")` defaults `0.0.0.0:8000`.

## Decision
New `Dockerfile` in the fork; build from the pinned commit; compose `build:`/image with pinned tag.

## Alternatives Considered
Prebuilt image (doesn't exist for the fork); venv-on-host (breaks puma-lan compose governance).

## Consequences
Image heavier than strictly needed (faster-whisper import deps) — accepted; `sherpa-onnx` wheel removed later by DEC-0016.
