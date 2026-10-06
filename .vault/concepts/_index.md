---
type: index
title: "Concepts"
createdAt: "2026-10-06T13:19:26Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: []
---

# Concepts

Domain glossary and mental models for transcription/diarization in this project.

## Nodes

- [[0001-diarization-clustering-length-degradation.concept]] — why sequential clustering fails on long audio (the root cause of the architecture)
- [[0002-llama-swap-generic-service-hosting.concept]] — hosting any OpenAI-compatible service via llama-swap (solver-visible GPU)
- [[0003-transcription-job-pipeline.concept]] — the job flow: compress → chunk → STT → diarize → align → export
- [[0004-speaker-segment-alignment.concept]] — joining diarization intervals onto STT segments
- [[0005-diarization-quality-eval-standard.concept]] — ground-truth-free quality gating (label distribution standard)
