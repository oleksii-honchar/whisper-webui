"""OpenAI-compatible Cloud Speech-to-Text Transcriber.

Universal, profile-driven adapter implementing the standard POST /audio/transcriptions
protocol. Seamlessly connects to Groq, OpenRouter, OpenAI, and any OpenAI-compatible STT endpoint.
Includes automatic audio compression and chunking for files exceeding the 25MB API limit.
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from transcribers.base import BaseTranscriber, Segment, TranscriptionResult, Word

logger = logging.getLogger(__name__)

# Standard OpenAI transcription endpoint limit is 25 MB; stay safely under
MAX_PAYLOAD_BYTES = 24 * 1024 * 1024  # 24 MB
CHUNK_DURATION_SECONDS = 1200  # 20 minutes for segmentation when file is very long


def _is_word_continuation(token: str) -> bool:
    """whisper tokenization convention: word-initial tokens carry a leading
    space, continuations of the same word do not. A token that starts with a
    letter and has no leading space continues the previous token's word."""
    return bool(token) and not token[0].isspace() and token[0].isalpha()


def _ends_with_word_char(token: str) -> bool:
    return bool(token) and token[-1].isalpha()


def _repair_mid_word_boundaries(segments: list[Segment]) -> None:
    """AH-14 Option A: no segment boundary may split a word.

    The engine (whisper.cpp) cuts segments at subword-token positions chosen
    by the model's timestamp tokens — ~2.7% of boundaries in the reference
    job cut a word in half ("мод|елей", "лок|ально", "агент|ство"). Detection
    is token-level (the engine's word tokens are authoritative — never a text
    heuristic): where the next segment's first token is a word continuation,
    the trailing fragment tokens are moved across the boundary to the segment
    where the word actually completes, and the boundary timestamps follow the
    moved word. Text-concatenation invariant: no text is lost, added, or
    reordered (whitespace may shift at a moved boundary — the fragment joins
    its word). Segments without word tokens (400-fallback path) are left
    untouched — there is no token evidence to act on.
    """
    i = 0
    while i + 1 < len(segments):
        cur, nxt = segments[i], segments[i + 1]
        if (
            not cur.words
            or not nxt.words
            or not _is_word_continuation(nxt.words[0].word)
            or not _ends_with_word_char(cur.words[-1].word)
        ):
            i += 1
            continue

        # Collect the split word's fragment tokens at the tail of cur: walk
        # back through continuation tokens to the word-initial token.
        frag: list[Word] = []
        while cur.words:
            tok = cur.words.pop()
            frag.insert(0, tok)
            if tok.word[:1].isspace():
                break  # word-initial token reached — fragment complete

        frag_text = "".join(t.word for t in frag).strip()
        cont_text = nxt.words[0].word.strip()
        if (
            not frag_text
            or not cur.text.rstrip().endswith(frag_text)
            or not nxt.text.startswith(cont_text)
        ):
            # Text and tokens disagree — restore the tokens, leave the
            # boundary untouched (conservative; invariant preserved).
            cur.words.extend(frag)
            i += 1
            continue

        cur.text = cur.text.rstrip()[: -len(frag_text)].rstrip()
        nxt.text = frag_text + nxt.text
        nxt.words = frag + list(nxt.words)
        # Timestamps follow the moved word: cur ends at its last complete
        # word, nxt starts no later than the moved token.
        if cur.words:
            cur.end = cur.words[-1].end
        nxt.start = min(nxt.start, frag[0].start)

        if not cur.words:
            # The word spanned the whole of cur — the segment is now empty.
            segments.pop(i)
            continue
        i += 1


class OpenAICompatibleTranscriber(BaseTranscriber):
    """Modular, configuration-driven audio transcriber for OpenAI-compatible REST APIs."""

    def __init__(
        self,
        name: str,
        display_name: str,
        base_url_getter: Callable[[], str] | str,
        api_key_getter: Callable[[], str | None] | str | None,
        default_model: str,
        supported_models: list[str],
        extra_headers: dict[str, str] | Callable[[], dict[str, str]] | None = None,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
    ):
        self.name = name
        self.display_name = display_name
        self._base_url_getter = base_url_getter
        self._api_key_getter = api_key_getter
        self.default_model = default_model
        self.supported_models = supported_models
        self._extra_headers = extra_headers or {}
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

    def get_api_key(self) -> str | None:
        """Dynamically resolve API key from callable or static string."""
        if callable(self._api_key_getter):
            return self._api_key_getter()
        return self._api_key_getter

    def get_base_url(self) -> str:
        """Dynamically resolve base URL and strip trailing slashes."""
        raw = self._base_url_getter() if callable(self._base_url_getter) else self._base_url_getter
        return (raw or "https://api.openai.com/v1").rstrip("/")

    def get_extra_headers(self) -> dict[str, str]:
        """Dynamically resolve extra headers."""
        if callable(self._extra_headers):
            return self._extra_headers()
        return dict(self._extra_headers)

    def is_available(self) -> bool:
        """Available if a valid API key is configured."""
        return bool(self.get_api_key())

    def _convert_numpy_to_wav(self, audio_array: Any, output_path: Path) -> Path:
        """Convert float32 NumPy audio array to 16kHz mono WAV file."""
        import wave
        import numpy as np

        int16_samples = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(int16_samples.tobytes())
        return output_path

    def _compress_to_mp3(self, input_path: Path, output_path: Path) -> Path:
        """Compress audio to 64kbps mono MP3 to optimize upload payload size."""
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(input_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "64k",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def _get_audio_duration(self, file_path: Path) -> float:
        """Probe audio duration in seconds using ffprobe."""
        cmd = [
            self.ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def _split_into_chunks(self, input_path: Path, temp_dir: Path) -> list[tuple[Path, float]]:
        """Split oversized audio into <= 20-minute chunks, returning list of (chunk_path, start_offset)."""
        pattern = str(temp_dir / "chunk_%03d.mp3")
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(input_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "64k",
            "-f", "segment",
            "-segment_time", str(CHUNK_DURATION_SECONDS),
            pattern,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        chunks: list[tuple[Path, float]] = []
        chunk_files = sorted(temp_dir.glob("chunk_*.mp3"))
        current_offset = 0.0
        for cf in chunk_files:
            dur = self._get_audio_duration(cf)
            chunks.append((cf, current_offset))
            current_offset += dur
        return chunks

    def _parse_verbose_json(
        self,
        data: dict[str, Any],
        time_offset: float = 0.0,
        start_segment_id: int = 0,
        on_segment: Callable[[Segment], None] | None = None,
    ) -> tuple[list[Segment], str, str, float]:
        """Parse OpenAI/Groq verbose_json response into standardized Segment & Word models."""
        raw_text = data.get("text", "")
        language = data.get("language", "auto")
        duration = float(data.get("duration", 0.0))

        raw_segments = data.get("segments", [])
        top_level_words = data.get("words", [])

        parsed_segments: list[Segment] = []

        if raw_segments:
            for idx, raw_seg in enumerate(raw_segments, start=start_segment_id):
                s_start = round(float(raw_seg.get("start", 0.0)) + time_offset, 3)
                s_end = round(float(raw_seg.get("end", 0.0)) + time_offset, 3)
                s_text = str(raw_seg.get("text", "")).strip()

                # Extract confidence from avg_logprob if available
                confidence = None
                if "avg_logprob" in raw_seg and raw_seg["avg_logprob"] is not None:
                    try:
                        confidence = round(math.exp(float(raw_seg["avg_logprob"])), 3)
                    except Exception:
                        confidence = None
                elif "confidence" in raw_seg and raw_seg["confidence"] is not None:
                    confidence = float(raw_seg["confidence"])

                # Word timestamps
                parsed_words: list[Word] = []
                seg_words = raw_seg.get("words", [])
                for w in seg_words:
                    w_start = round(float(w.get("start", 0.0)) + time_offset, 3)
                    w_end = round(float(w.get("end", 0.0)) + time_offset, 3)
                    parsed_words.append(
                        Word(
                            word=str(w.get("word", "")),
                            start=w_start,
                            end=w_end,
                            probability=w.get("probability"),
                        )
                    )

                seg_obj = Segment(
                    id=idx,
                    start=s_start,
                    end=s_end,
                    text=s_text,
                    confidence=confidence,
                    words=parsed_words,
                )
                parsed_segments.append(seg_obj)
                if on_segment:
                    on_segment(seg_obj)

            # If top-level words exist but segments lacked words, assign them to segments
            if top_level_words and all(len(s.words) == 0 for s in parsed_segments):
                for tw in top_level_words:
                    tw_start = round(float(tw.get("start", 0.0)) + time_offset, 3)
                    tw_end = round(float(tw.get("end", 0.0)) + time_offset, 3)
                    word_obj = Word(
                        word=str(tw.get("word", "")),
                        start=tw_start,
                        end=tw_end,
                        probability=tw.get("probability"),
                    )
                    for seg in parsed_segments:
                        if seg.start <= tw_start <= seg.end:
                            seg.words.append(word_obj)
                            break

            # AH-14 Option A: repair mid-word boundaries after word tokens
            # have been assigned (live SSE already streamed the engine's
            # segments via on_segment above — the final result is repaired).
            _repair_mid_word_boundaries(parsed_segments)
        else:
            # Fallback when provider returns text without segmented timestamps
            if raw_text.strip():
                seg_obj = Segment(
                    id=start_segment_id,
                    start=round(time_offset, 3),
                    end=round(time_offset + duration, 3),
                    text=raw_text.strip(),
                )
                parsed_segments.append(seg_obj)
                if on_segment:
                    on_segment(seg_obj)

        return parsed_segments, raw_text, language, duration

    def _call_transcription_api(
        self,
        file_path: Path,
        model_name: str,
        language: str | None = None,
        request_word_timestamps: bool = True,
    ) -> dict[str, Any]:
        """Execute HTTP POST request to {base_url}/audio/transcriptions."""
        api_key = self.get_api_key()
        if not api_key:
            raise RuntimeError(f"{self.display_name} API key is not configured.")

        url = f"{self.get_base_url()}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            **self.get_extra_headers(),
        }

        form_data: list[tuple[str, str]] = [
            ("model", model_name),
            ("response_format", "verbose_json"),
            ("temperature", "0.0"),
        ]

        if language and language != "auto":
            form_data.append(("language", language))

        if request_word_timestamps:
            form_data.append(("timestamp_granularities[]", "word"))
            form_data.append(("timestamp_granularities[]", "segment"))

        content_type = "audio/mpeg" if file_path.suffix.lower() == ".mp3" else "audio/wav"

        from config import settings
        timeout = httpx.Timeout(float(settings.stt_request_timeout), connect=15.0)
        with open(file_path, "rb") as f:
            # httpx>=0.28 cannot encode data=list-of-tuples together with files=
            # (TypeError in the multipart encoder). Send the form fields through the
            # same multipart list: a (None, value) tuple renders as a plain field
            # and keeps repeated keys like timestamp_granularities[].
            files: list[tuple[str, tuple]] = [("file", (file_path.name, f, content_type))]
            files += [(name, (None, value)) for name, value in form_data]
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, files=files)

        # Graceful fallback: some providers fail if timestamp_granularities[] is passed
        if resp.status_code == 400 and request_word_timestamps:
            err_text = resp.text.lower()
            if "timestamp_granularities" in err_text or "granularit" in err_text or "parameter" in err_text:
                logger.info("Provider does not support timestamp_granularities[], retrying without it.")
                return self._call_transcription_api(
                    file_path=file_path,
                    model_name=model_name,
                    language=language,
                    request_word_timestamps=False,
                )

        if resp.status_code != 200:
            raise RuntimeError(
                f"{self.display_name} API error (HTTP {resp.status_code}): {resp.text}"
            )

        return resp.json()

    def transcribe(
        self,
        audio: Path | Any,
        model_name: str | None = None,
        language: str | None = None,
        vad_filter: bool = True,
        on_segment: Callable[[Segment], None] | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Perform audio transcription via OpenAI-compatible REST API.

        Optional kwargs:
        - on_progress(done_audio_s, total_audio_s, phase): progress callback,
          phase in {"compressing", "chunk"}; cumulative audio-seconds done.

        Automatically handles:
        - In-memory NumPy audio conversion
        - FFmpeg audio compression for payloads exceeding 24MB
        - Audio chunking and timestamp re-alignment for extended audio (>50 minutes)
        - Verbose JSON response parsing into Segment & Word models
        """
        if not self.is_available():
            raise RuntimeError(f"{self.display_name} API key is not configured. Please set in Settings.")

        chosen_model = model_name or self.default_model
        on_progress = kwargs.get("on_progress")

        def _emit_progress(done_audio_s: float, total_audio_s: float, phase: str) -> None:
            if on_progress is not None:
                on_progress(done_audio_s, total_audio_s, phase)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            target_file: Path

            # 1. Handle in-memory NumPy array
            if hasattr(audio, "__array__") or type(audio).__name__ == "ndarray":
                target_file = temp_dir / "converted_audio.wav"
                self._convert_numpy_to_wav(audio, target_file)
            else:
                target_file = Path(audio).resolve()
                if not target_file.is_file():
                    raise FileNotFoundError(f"Audio file not found: {target_file}")

            # 2. Check file size. If > 24MB or not an audio format, compress to 64kbps MP3
            file_size = target_file.stat().st_size
            audio_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

            if file_size > MAX_PAYLOAD_BYTES or target_file.suffix.lower() not in audio_extensions:
                compressed_file = temp_dir / "compressed_audio.mp3"
                logger.info(
                    "Input file size (%d bytes) or format (%s) requires compression for %s.",
                    file_size,
                    target_file.suffix,
                    self.display_name,
                )
                # Total audio duration is not known until the file is probed; emit a
                # message-only "compressing" signal (done=0, total=0) before compressing.
                _emit_progress(0.0, 0.0, "compressing")
                self._compress_to_mp3(target_file, compressed_file)
                target_file = compressed_file
                file_size = target_file.stat().st_size

            # 3. If still > 24MB (e.g. audio longer than ~50 minutes), chunk into slices
            if file_size > MAX_PAYLOAD_BYTES:
                logger.info("Compressed audio still exceeds 24MB; chunking into segments...")
                chunks = self._split_into_chunks(target_file, temp_dir)
            else:
                chunks = [(target_file, 0.0)]

            # Probe per-chunk audio durations for honest progress (cumulative audio-seconds).
            # Only probed when a progress callback is present to avoid extra ffprobe calls.
            if on_progress is not None:
                chunk_durations = [self._get_audio_duration(cp) for cp, _ in chunks]
                total_audio_s = sum(chunk_durations)
            else:
                chunk_durations = [0.0] * len(chunks)
                total_audio_s = 0.0

            # 4. Transcribe chunk(s) and aggregate results
            all_segments: list[Segment] = []
            combined_texts: list[str] = []
            detected_language = "auto"
            total_duration = 0.0
            next_seg_id = 0
            done_audio_s = 0.0
            num_chunks = len(chunks)

            for idx, ((chunk_path, time_offset), chunk_audio_dur) in enumerate(
                zip(chunks, chunk_durations), start=1
            ):
                api_start = time.time()
                data = self._call_transcription_api(
                    file_path=chunk_path,
                    model_name=chosen_model,
                    language=language,
                    request_word_timestamps=True,
                )
                api_latency = time.time() - api_start

                chunk_segs, chunk_text, chunk_lang, chunk_dur = self._parse_verbose_json(
                    data=data,
                    time_offset=time_offset,
                    start_segment_id=next_seg_id,
                    on_segment=on_segment,
                )

                all_segments.extend(chunk_segs)
                next_seg_id += len(chunk_segs)
                if chunk_text.strip():
                    combined_texts.append(chunk_text.strip())
                if chunk_lang and chunk_lang != "auto":
                    detected_language = chunk_lang
                total_duration += chunk_dur

                done_audio_s += chunk_audio_dur
                logger.info(
                    "Transcribed chunk %d/%d (offset=%.1fs, api_latency=%.2fs)",
                    idx,
                    num_chunks,
                    time_offset,
                    api_latency,
                )
                _emit_progress(done_audio_s, total_audio_s, "chunk")

            full_text = " ".join(combined_texts).strip()

            return TranscriptionResult(
                text=full_text,
                segments=all_segments,
                language=detected_language,
                duration=round(total_duration, 2),
            )

    def test_connection(self) -> dict[str, Any]:
        """Test API authentication and latency against the provider."""
        api_key = self.get_api_key()
        if not api_key:
            return {
                "online": False,
                "base_url": self.get_base_url(),
                "error": f"{self.display_name} API Key is not configured.",
            }

        url = f"{self.get_base_url()}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            **self.get_extra_headers(),
        }

        start_time = time.time()
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(url, headers=headers)
                latency_ms = round((time.time() - start_time) * 1000, 1)
                return {
                    "online": resp.status_code == 200,
                    "status_code": resp.status_code,
                    "latency_ms": latency_ms,
                    "base_url": self.get_base_url(),
                    "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}: {resp.text[:120]}",
                }
        except Exception as e:
            return {
                "online": False,
                "base_url": self.get_base_url(),
                "error": str(e),
            }
