---
type: decision
id: DEC-0002
title: "Service name: transcriber (not whisper-webui)"
status: accepted
createdAt: "2026-10-06T14:37:21Z"
updatedAt: "2026-10-06T14:37:21Z"
tags: [naming, deployment]
system: transcriber
see_also: ["specifications/0001-transcriber-service.spec.md"]
---

# DEC-0002: Service name: transcriber (not whisper-webui)

## Context
The fork repo is named `whisper-webui`, but the app identity is "Audio Transcriber" (jeckyllX/transcriber); `whisper-webui` collides with a superseded jhj0517/Whisper-WebUI plan and implies in-container whisper.

## Decision
Service, Caddy route (`transcriber.lan`), Infisical path (`/transcriber`), puma-lan folder and docs all use **transcriber**.

## Alternatives Considered
`whisper-webui` — matches the fork repo name but ambiguous and implies local whisper.

## Consequences
Unambiguous homelab identity; fork repo rename to `transcriber` optional (open decision, not required).
