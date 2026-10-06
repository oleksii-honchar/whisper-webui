---
type: runbook
title: "Deploy a new transcriber image (build → push → pin → deploy)"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [deployment, images]
system: transcriber
see_also: ["decisions/0007-dockerfile-for-fork.decision.md", "architectures/transcriber/containers/0001-transcriber.container.md"]
---

# Runbook: transcriber image chain deploy

## Prerequisites
Fork checkout at the target commit; Docker Hub push access; `puma-lan` repo access.

## Steps
1. `./build.sh` at the pinned commit → builds + pushes `tuiteraz/whisper-webui:<sha>` and `:latest`.
2. Update `puma-lan/transcriber/docker-compose.yaml` image pin to `<sha>` (recommended — ⚠️ current compose uses `:latest`; pin policy is an open decision).
3. Push puma-lan; on puma: `./restart.sh` in `puma-lan/transcriber/` (Infisical-wrapped compose).

## Verification
- `https://transcriber.lan` loads; `GET /api/status` shows `diarization_engines: [remote]` available.
- Run a short multi-speaker job; confirm transcript + speaker labels + exports.

## Rollback
Revert the compose image pin to the previous sha and restart. (Since DEC-0016 rollback is image-pin revert, not an env flip.)
