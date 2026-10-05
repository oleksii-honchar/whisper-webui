"""Remote speaker diarization engine — HTTP client for a /diarize sidecar.

AH-11b (DEC-14): talks to the Nemotron-3 sidecar (directly or through the
llama-swap /upstream passthrough) at {DIARIZATION_API_URL}/diarize. The
sidecar stays raw/faithful to model output; the de-blip policy (DEC-15, R2')
lives HERE, on the integration side, so it applies regardless of who serves
/diarize and never touches the sherpa path.
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from config import settings
from diarization.base import BaseDiarizer, DiarizationResult, SpeakerInterval

logger = logging.getLogger(__name__)


class RemoteDiarizer(BaseDiarizer):
    """Speaker diarization via an HTTP sidecar exposing POST /diarize.

    Contract (spec §5, rev3): multipart ``file`` upload (WAV/mp3) in,
    ``{num_speakers, intervals:[{start, end, speaker}]}`` out — mapped 1:1
    onto DiarizationResult/SpeakerInterval.

    ``num_speakers`` and ``cluster_threshold`` are ACCEPTED AND IGNORED: the
    upstream model is end-to-end (no clustering stage, arrival-ordered ≤8
    speakers) — these sherpa-path parameters have no meaning here (DEC-14).

    ``is_available()`` is deliberately config-presence only — NO health probe:
    a probe through llama-swap's /upstream would trigger a model load on
    every check.

    ``use_process_pool = False`` (R3): an HTTP call has no GIL problem and
    must not pickle ~183 MB audio samples into a child process — the thread
    executor path in run_pipeline is used instead.
    """

    name: str = "remote"
    display_name: str = "Remote Diarizer (HTTP)"
    use_process_pool: bool = False

    def __init__(self, transport: httpx.BaseTransport | None = None):
        # Test seam: unit tests inject httpx.MockTransport (no live sidecar).
        # Production leaves this None → httpx's default transport.
        self._transport = transport

    def is_available(self) -> bool:
        """Available when a diarization API URL is configured (no health probe)."""
        return bool(settings.diarization_api_url)

    def _get_base_url(self) -> str:
        """Resolve the configured sidecar base URL, stripping trailing slashes."""
        return (settings.diarization_api_url or "").rstrip("/")

    @staticmethod
    def _samples_to_wav_bytes(audio: Any) -> bytes:
        """float32 mono 16 kHz samples → in-memory WAV via stdlib wave + int16
        (same conversion as openai_compat._convert_numpy_to_wav — no new deps)."""
        int16_samples = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(int16_samples.tobytes())
        return buf.getvalue()

    def _prepare_upload(self, audio: Path | Any) -> tuple[str, bytes, str]:
        """Path → the file's own bytes; float32 ndarray → in-memory WAV."""
        if isinstance(audio, Path):
            return audio.name, audio.read_bytes(), "application/octet-stream"
        return "audio.wav", self._samples_to_wav_bytes(audio), "audio/wav"

    def diarize(
        self,
        audio: Path | Any,
        num_speakers: int = -1,
        cluster_threshold: float = 0.5,
        **kwargs: Any,
    ) -> DiarizationResult:
        """POST multipart audio to {api_url}/diarize, apply the de-blip filter
        (R2'), and return the re-labeled DiarizationResult.

        num_speakers/cluster_threshold: accepted-and-ignored (see class docstring).
        """
        filename, payload, content_type = self._prepare_upload(audio)
        timeout = httpx.Timeout(float(settings.diarization_request_timeout), connect=15.0)
        with httpx.Client(timeout=timeout, transport=self._transport) as client:
            response = client.post(
                f"{self._get_base_url()}/diarize",
                files={"file": (filename, payload, content_type)},
            )
            response.raise_for_status()
            data = response.json()

        intervals = [
            SpeakerInterval(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker=str(item["speaker"]),
            )
            for item in data.get("intervals", [])
        ]
        return self._apply_min_speaker_duration(intervals, int(data.get("num_speakers", 0)))

    def _apply_min_speaker_duration(
        self, intervals: list[SpeakerInterval], raw_num_speakers: int
    ) -> DiarizationResult:
        """De-blip policy (R2', DEC-15): drop every speaker whose TOTAL interval
        duration is below settings.diarization_min_speaker_duration, re-label
        survivors in arrival order (spk0, spk1, …), recompute num_speakers.

        Per-SPEAKER, not per-interval — short turns of real speakers are
        legitimate; the artifact this targets is a spurious speaker.
        Floor 0 disables the filter entirely (raw labels preserved).
        """
        floor = float(settings.diarization_min_speaker_duration)
        if floor <= 0:
            return DiarizationResult(num_speakers=raw_num_speakers, intervals=intervals)

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for interval in intervals:
            spk = interval.speaker
            totals[spk] = totals.get(spk, 0.0) + (interval.end - interval.start)
            counts[spk] = counts.get(spk, 0) + 1

        # Arrival order of SURVIVING speakers (first appearance in the interval list).
        kept: list[str] = []
        for interval in intervals:
            spk = interval.speaker
            if totals[spk] >= floor and spk not in kept:
                kept.append(spk)

        for spk in totals:
            if totals[spk] < floor:
                logger.info(
                    "Diarization de-blip: dropped speaker '%s' (total %.2fs < %.2fs floor, %d interval(s))",
                    spk,
                    totals[spk],
                    floor,
                    counts[spk],
                )

        remap = {old: f"spk{idx}" for idx, old in enumerate(kept)}
        kept_intervals = [
            SpeakerInterval(start=i.start, end=i.end, speaker=remap[i.speaker])
            for i in intervals
            if i.speaker in remap
        ]
        return DiarizationResult(num_speakers=len(kept), intervals=kept_intervals)
