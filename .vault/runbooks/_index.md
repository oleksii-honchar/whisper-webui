---
type: index
title: "Runbooks"
createdAt: "2026-10-06T13:19:26Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: []
---

# Runbooks

Operational procedures (build, deploy, verify, rollback) for transcriber on puma.lan.

## Nodes

- [[0001-transcriber-image-chain-deploy.runbook]] — build → push → pin → deploy the app image
- [[0002-diarizer-sidecar-lifecycle-check.runbook]] — cold-start / health / eviction / VRAM check via llama-swap
- [[0003-diarizer-sidecar-image-build.runbook]] — build and roll the nemotron3-diarizer image
