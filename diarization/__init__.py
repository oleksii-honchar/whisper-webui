"""Speaker diarization module."""

from diarization.base import BaseDiarizer, DiarizationResult, SpeakerInterval
from diarization.alignment import align_speakers_to_segments
from diarization.factory import diarizer_factory

__all__ = [
    "BaseDiarizer",
    "DiarizationResult",
    "SpeakerInterval",
    "align_speakers_to_segments",
    "diarizer_factory",
]
