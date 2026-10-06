"""Asynchronous Background Job Queue & Event Streaming.

Allows long audio files to be processed without blocking HTTP connections,
streaming real-time progress and transcribed segments via Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from audio_processor import AudioProcessor
from notifications import dispatcher, NotificationPayload
from transcribers import transcriber_factory
from transcribers.base import Segment
from llm import llm_registry, get_polish_prompt, ChunkedSummarizer

logger = logging.getLogger(__name__)

# Heartbeat interval for the diarization phase (no sub-progress available from
# sherpa's blocking process() — report elapsed seconds instead, never fake %).
DIARIZATION_HEARTBEAT_SECONDS = 30.0


def _diarize_worker(audio: Any, num_speakers: int, cluster_threshold: float, engine: str):
    """Process-pool worker for the diarization phase (P13, DEC-9; R4/DEC-14).

    Module-level so ProcessPoolExecutor can pickle it by reference into the
    child; the diarizer is constructed IN-CHILD from the ENGINE NAME passed by
    run_pipeline (resolved from settings there — never a hardcoded constant)
    and the blocking sherpa process() runs there — off the event loop's GIL,
    so the UI/SSE feed stays responsive while diarization runs. Model download
    stays in the parent (run_pipeline calls ensure_models before submitting).
    """
    from diarization import diarizer_factory

    diarizer = diarizer_factory.get_diarizer(engine)
    return diarizer.diarize(audio, num_speakers=num_speakers, cluster_threshold=cluster_threshold)


@dataclass
class Job:
    """Represents a background audio processing task."""
    job_id: str
    filename: str
    status: str = "queued"  # queued, converting, transcribing, processing_ai, completed, failed
    progress: int = 0
    message: str = "Job queued"
    duration: float = 0.0
    processing_time: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_queues: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)


class JobManager:
    """In-memory job coordinator with live event fan-out."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            from config import settings
            self._semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        return self._semaphore

    def create_job(self, filename: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, filename=filename)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def rename_speaker(self, job_id: str, old_name: str, new_name: str) -> dict[str, Any] | None:
        """Rename a detected speaker across all segments, turns, and subtitle exports."""
        job = self.get_job(job_id)
        if not job or not job.result:
            return None

        segments_data = job.result.get("segments", [])
        for s in segments_data:
            if s.get("speaker") == old_name:
                s["speaker"] = new_name
            for w in s.get("words", []):
                if w.get("speaker") == old_name:
                    w["speaker"] = new_name

        from transcribers.base import Segment, TranscriptionResult
        segments = [Segment(**s) for s in segments_data]
        res = TranscriptionResult(
            text=job.result.get("text", ""),
            segments=segments,
            language=job.result.get("language", "auto"),
            duration=job.result.get("duration", 0.0),
        )
        job.result["text"] = res.to_txt()
        job.result["srt"] = res.to_srt()
        job.result["vtt"] = res.to_vtt()
        job.result["ass"] = res.to_ass()
        job.result["word_vtt"] = res.to_word_vtt()
        job.result["segments"] = [s.model_dump() for s in segments]
        job.result["speaker_turns"] = res.get_speaker_turns()

        self.emit(job_id, "speaker_renamed", {
            "old_name": old_name,
            "new_name": new_name,
            "speaker_turns": job.result["speaker_turns"],
        })
        return job.result

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found")
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job.event_queues.append(q)
        return q

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        job = self.get_job(job_id)
        if job and queue in job.event_queues:
            job.event_queues.remove(queue)

    def emit(self, job_id: str, event: str, data: dict[str, Any]) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        payload = {"event": event, "data": data, "timestamp": time.time()}
        for q in list(job.event_queues):
            try:
                q.put_nowait(payload)
            except Exception:
                pass

    def update_status(self, job_id: str, status: str, progress: int, message: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.status = status
        job.progress = progress
        job.message = message
        self.emit(job_id, "status", {"status": status, "progress": progress, "message": message})

    async def _diarization_heartbeat(self, job_id: str) -> None:
        """Re-emit the diarizing status with elapsed seconds while the blocking
        sherpa process() is pending. No fake percentages — sherpa exposes no
        sub-progress; elapsed time is the only honest signal."""
        started = time.monotonic()
        while True:
            await asyncio.sleep(DIARIZATION_HEARTBEAT_SECONDS)
            elapsed = int(time.monotonic() - started)
            self.update_status(job_id, "diarizing", 60, f"Diarizing… {elapsed}s elapsed")

    async def run_pipeline(
        self,
        job_id: str,
        media_path: Path,
        whisper_engine: str = "faster-whisper",
        whisper_model: str = "base",
        language: str | None = None,
        vad_filter: bool = True,
        enable_diarization: bool = False,
        num_speakers: int = -1,
        cluster_threshold: float | None = None,
        ai_action: str = "summary",  # raw, polish, summary
        summary_level: str = "bullets",
        llm_provider: str = "ollama",
        llm_model: str | None = None,
        notify: bool = False,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        # The clustering threshold's default is settings.diarization_threshold —
        # the setting is live end-to-end, no hardcoded constant (P14, DEC-10).
        from config import settings
        if cluster_threshold is None:
            cluster_threshold = settings.diarization_threshold

        start_time = time.time()
        audio_processor = AudioProcessor()
        converted_wav: Path | None = None

        # AH-13: resolve the diarization engine at job start so the job-start INFO
        # line names it — the wrong-engine path was invisible in logs
        # (evidence-T10E2E-env.txt: only the STT engine was named). is_available()
        # is a config check; the resolved instance is reused below (no re-resolve).
        diarizer = None
        diarization_available = False
        if enable_diarization:
            try:
                from diarization import diarizer_factory
                diarizer = diarizer_factory.get_diarizer(settings.diarization_engine)
                diarization_available = diarizer.is_available()
            except Exception as diar_init_err:
                logger.warning(
                    "Diarization engine %s could not be initialized: %s",
                    settings.diarization_engine, diar_init_err,
                )

        try:
            # Queue waits become visible: the Job already models "queued" — emit it
            # BEFORE acquiring the semaphore so saturated queues are observable (P6).
            self.update_status(job_id, "queued", 5, "Waiting in queue...")
            # AH-13 (P5 extension): one INFO line at job start, now also carrying
            # the resolved diarization engine + availability.
            diarization_note = (
                f" diarization_engine={settings.diarization_engine} available={diarization_available}"
                if enable_diarization
                else ""
            )
            logger.info(
                "Job %s started: engine=%s model=%s language=%s%s",
                job_id, whisper_engine, whisper_model, language or "auto", diarization_note,
            )

            async with self.semaphore:
                # 1. Convert audio
                self.update_status(job_id, "converting", 15, "Decoding audio with FFmpeg...")
                convert_start = time.time()
                transcriber = transcriber_factory.get_transcriber(whisper_engine)

                from audio_processor import HAS_NUMPY

                use_memory = settings.use_in_memory_pcm and HAS_NUMPY and whisper_engine == "faster-whisper"

                if use_memory:
                    audio_input, metadata = audio_processor.convert_to_pcm_array(media_path)
                else:
                    converted_wav = media_path.parent / f"{job_id}_16k.wav"
                    audio_input = audio_processor.convert_to_whisper_wav(media_path, converted_wav)
                    metadata = audio_processor.probe_media(media_path)

                job.duration = metadata.duration
                logger.info(
                    "Job %s: audio converted in %.2fs (duration=%.1fs)",
                    job_id, time.time() - convert_start, job.duration,
                )

                # 2. Transcribe with live segment streaming
                self.update_status(job_id, "transcribing", 35, "Transcribing with Whisper...")

                def on_segment_callback(seg: Segment):
                    self.emit(job_id, "segment", seg.model_dump())

                def on_progress_callback(done_audio_s: float, total_audio_s: float, phase: str) -> None:
                    # Map adapter progress into the 35→50 band; message-only when the
                    # total audio duration is not (yet) known — no invented percents.
                    if phase == "compressing":
                        self.update_status(job_id, "transcribing", 35, "Compressing audio for upload...")
                        return
                    if total_audio_s > 0:
                        progress = min(50, 35 + int(15 * done_audio_s / total_audio_s))
                        self.update_status(
                            job_id,
                            "transcribing",
                            progress,
                            f"Transcribing audio... {done_audio_s:.0f}/{total_audio_s:.0f}s",
                        )
                    else:
                        self.update_status(job_id, "transcribing", 35, "Transcribing audio...")

                # Run CPU-bound transcription in threadpool to avoid blocking event loop
                loop = asyncio.get_running_loop()
                transcribe_start = time.time()
                transcription_result = await loop.run_in_executor(
                    None,
                    lambda: transcriber.transcribe(
                        audio_input,
                        model_name=whisper_model,
                        language=language if language != "auto" else None,
                        vad_filter=vad_filter,
                        on_segment=on_segment_callback,
                        on_progress=on_progress_callback,
                    ),
                )
                transcribe_elapsed = time.time() - transcribe_start
                logger.info(
                    "Job %s: transcription done in %.2fs: %d segments, RTF %.3f",
                    job_id,
                    transcribe_elapsed,
                    len(transcription_result.segments),
                    transcribe_elapsed / job.duration if job.duration > 0 else 0.0,
                )

                # 2.5 Speaker Diarization
                num_speakers_detected = 0
                if enable_diarization:
                    try:
                        from diarization import align_speakers_to_segments
                        # R4 (DEC-14): engine resolved from settings — rollback is
                        # one env var (DIARIZATION_ENGINE=sherpa-onnx). AH-13: the
                        # diarizer was already resolved at job start (same
                        # settings.diarization_engine); reuse it — no re-resolve.
                        if diarizer is not None and diarization_available:
                            if hasattr(diarizer, "models_ready") and not diarizer.models_ready():
                                self.update_status(job_id, "diarizing", 52, "Downloading speaker diarization models on demand (first run only)...")
                                # Model download stays in the parent (thread pool —
                                # network I/O): the process-pool child must never
                                # re-download (P13).
                                await loop.run_in_executor(None, diarizer.ensure_models)
                            else:
                                self.update_status(job_id, "diarizing", 60, "Identifying speakers (diarization)...")
                            heartbeat_task = asyncio.create_task(self._diarization_heartbeat(job_id))
                            diarize_start = time.time()
                            try:
                                if diarizer.use_process_pool:
                                    # P13 (DEC-9): the blocking sherpa process() runs in a
                                    # child process — the GIL no longer freezes the UI/SSE
                                    # feed while diarization runs. Heartbeat keeps covering
                                    # the pool await (try/finally preserved from T6).
                                    with ProcessPoolExecutor(max_workers=1) as pool:
                                        diar_result = await loop.run_in_executor(
                                            pool,
                                            _diarize_worker,
                                            audio_input,
                                            num_speakers,
                                            cluster_threshold,
                                            settings.diarization_engine,
                                        )
                                else:
                                    # R4 (DEC-14): HTTP-backed engines (use_process_pool
                                    # False) run in the thread executor — same pattern as
                                    # transcribe above. An HTTP call has no GIL problem,
                                    # and ~183 MB audio samples are never pickled into a
                                    # child process.
                                    diar_result = await loop.run_in_executor(
                                        None,
                                        lambda: diarizer.diarize(
                                            audio_input,
                                            num_speakers=num_speakers,
                                            cluster_threshold=cluster_threshold,
                                        ),
                                    )
                            finally:
                                heartbeat_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await heartbeat_task
                            align_speakers_to_segments(transcription_result.segments, diar_result)
                            num_speakers_detected = diar_result.num_speakers
                            diarize_elapsed = time.time() - diarize_start
                            logger.info(
                                "Job %s: diarization done in %.2fs: RTF %.3f, %d speakers",
                                job_id,
                                diarize_elapsed,
                                diarize_elapsed / job.duration if job.duration > 0 else 0.0,
                                num_speakers_detected,
                            )
                            self.emit(job_id, "diarization", {
                                "num_speakers": diar_result.num_speakers,
                                "speakers": diar_result.speaker_names,
                                "intervals": [i.model_dump() for i in diar_result.intervals],
                            })
                        else:
                            # AH-13: the skip warning names the engine it skipped.
                            logger.warning(
                                "Speaker diarization requested but engine %s is not available; skipping.",
                                settings.diarization_engine,
                            )
                    except Exception as diar_err:
                        logger.warning("Diarization failed for job %s: %s", job_id, diar_err)

            job_result: dict[str, Any] = {
                "task_id": job_id,
                "filename": job.filename,
                "duration": job.duration or transcription_result.duration,
                "language": transcription_result.language,
                "engine_used": transcriber.name,
                "model_used": whisper_model,
                "num_speakers": num_speakers_detected,
                "speaker_turns": transcription_result.get_speaker_turns(),
                "text": transcription_result.to_txt(),
                "srt": transcription_result.to_srt(),
                "vtt": transcription_result.to_vtt(),
                "ass": transcription_result.to_ass(),
                "word_vtt": transcription_result.to_word_vtt(),
                "segments": [s.model_dump() for s in transcription_result.segments],
                "polished": None,
                "summary": None,
            }

            # 3. AI Post-Processing (gracefully degraded if LLM provider is offline)
            if ai_action in ("polish", "summary") and transcription_result.text.strip():
                self.update_status(job_id, "processing_ai", 75, f"Running AI {ai_action}...")
                try:
                    provider = llm_registry.get_provider(llm_provider)

                    if ai_action == "polish":
                        sys_prompt, user_prompt = get_polish_prompt(transcription_result.text)
                        accumulated = []
                        async for token in provider.generate_stream(
                            prompt=user_prompt, system_prompt=sys_prompt, model=llm_model
                        ):
                            accumulated.append(token)
                            self.emit(job_id, "ai_token", {"token": token, "action": "polish"})
                        job_result["polished"] = "".join(accumulated)

                    elif ai_action == "summary":
                        summarizer = ChunkedSummarizer(provider)
                        accumulated = []
                        async for token in summarizer.summarize_stream(
                            transcript=transcription_result.text,
                            level=summary_level,
                            model=llm_model,
                        ):
                            accumulated.append(token)
                            self.emit(job_id, "ai_token", {"token": token, "action": "summary"})
                        job_result["summary"] = "".join(accumulated)
                except Exception as ai_err:
                    logger.warning("AI post-processing (%s) failed for job %s: %s", ai_action, job_id, ai_err)
                    warn_text = f"AI {ai_action} skipped: {ai_err}. Check that Ollama is running ('ollama serve') or your API key is valid."
                    job_result["ai_warning"] = warn_text
                    if ai_action == "summary":
                        job_result["summary"] = f"[{warn_text}]"
                    elif ai_action == "polish":
                        job_result["polished"] = f"[{warn_text}]"
                    self.emit(job_id, "status", {"state": "warning", "progress": 85, "message": warn_text})

            # 4. Optional Notification Dispatch
            if notify:
                self.update_status(job_id, "dispatching_notification", 95, "Dispatching notifications...")
                payload = NotificationPayload(
                    task_id=job_id,
                    filename=job.filename,
                    duration_seconds=job_result["duration"],
                    processing_time_seconds=round(time.time() - start_time, 2),
                    whisper_model=whisper_model,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    summary_type=ai_action,
                    transcript_text=job_result["text"],
                    summary_text=job_result["summary"],
                    polished_text=job_result["polished"],
                )
                await dispatcher.dispatch_all(payload)

            elapsed = round(time.time() - start_time, 2)
            job.processing_time = elapsed
            job_result["processing_time"] = elapsed
            job.result = job_result

            logger.info("Job %s completed in %.2fs", job_id, elapsed)
            self.update_status(job_id, "completed", 100, f"Completed in {elapsed}s")
            self.emit(job_id, "completed", job_result)

        except Exception as e:
            logger.exception("Pipeline job %s failed", job_id)
            job.status = "failed"
            job.error = str(e)
            self.update_status(job_id, "failed", 0, str(e))
            self.emit(job_id, "failed", {"error": str(e)})

        finally:
            if converted_wav and converted_wav.exists():
                try:
                    converted_wav.unlink()
                except Exception:
                    pass
            # Clean up source file
            if media_path.exists():
                try:
                    media_path.unlink()
                except Exception:
                    pass


# Global job manager instance
job_manager = JobManager()
