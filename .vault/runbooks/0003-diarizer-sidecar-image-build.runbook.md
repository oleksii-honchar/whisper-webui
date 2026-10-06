---
type: runbook
title: "Build and roll the nemotron3-diarizer sidecar image"
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [diarization, images, build]
system: transcriber
see_also: ["architectures/transcriber/containers/0002-nemotron3-diarizer-sidecar.container.md", "runbooks/0002-diarizer-sidecar-lifecycle-check.runbook.md"]
---

# Runbook: diarizer sidecar image build

## Prerequisites
Fork checkout with `diarizer-service/` (`app.py` + `Dockerfile`); Docker build access; model download at build time (~0.4 GB baked into the image).

## Steps
1. `docker build -t nemotron3-diarizer:<tag> diarizer-service/` (NeMo `nemo_toolkit[asr]` + ffmpeg; model baked at build).
2. Transfer/build on puma; the llama-swap config entry references the tag (`cmd: docker run … nemotron3-diarizer:<tag>`).
3. Update the tag in `puma-lan/llama-swap/configs/config-1.yaml`; llama-swap hot-reloads config — no service restart.

## Verification
Run the lifecycle check: [[runbooks/0002-diarizer-sidecar-lifecycle-check]] (cold start via /upstream, health, VRAM within the recorded baseline).

## Rollback
Point the config entry back to the previous tag (hot-reload); old image stays available.
