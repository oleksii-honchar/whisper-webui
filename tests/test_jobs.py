"""Tests for Background Job Manager and streaming event queues."""

import asyncio
import contextlib
import functools
import logging
import os
import re
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from diarization.base import DiarizationResult, SpeakerInterval
from jobs import JobManager, Job
from transcribers.base import Segment, TranscriptionResult


def test_job_lifecycle():
    manager = JobManager()
    job = manager.create_job("test_audio.mp3")

    assert job.job_id is not None
    assert job.filename == "test_audio.mp3"
    assert job.status == "queued"
    assert manager.get_job(job.job_id) is job

    # Status update
    manager.update_status(job.job_id, "converting", 25, "Decoding audio...")
    assert job.status == "converting"
    assert job.progress == 25
    assert job.message == "Decoding audio..."


@pytest.mark.asyncio
async def test_job_event_subscription():
    manager = JobManager()
    job = manager.create_job("test_stream.wav")

    queue = manager.subscribe(job.job_id)
    assert len(job.event_queues) == 1

    manager.emit(job.job_id, "segment", {"text": "Hello world", "start": 0.0, "end": 2.0})
    event = await queue.get()

    assert event["event"] == "segment"
    assert event["data"]["text"] == "Hello world"

    manager.unsubscribe(job.job_id, queue)
    assert len(job.event_queues) == 0


# ---------------------------------------------------------------------------
# T6 — AH-4 observability (spec §3.4 P4-P9, DEC-8).
# Tests assert status-event sequences and payloads — never logger calls —
# except where the log line IS the observable contract (P5/P9 stage durations,
# asserted via caplog.records content per spec success criterion 6).
# ---------------------------------------------------------------------------

AUDIO_DURATION_S = 120.0


def _fake_transcription_result(n_segments: int = 2) -> TranscriptionResult:
    segments = [
        Segment(id=i, start=i * 10.0, end=i * 10.0 + 5.0, text=f"segment {i}")
        for i in range(n_segments)
    ]
    return TranscriptionResult(
        text="hello world",
        segments=segments,
        language="en",
        duration=AUDIO_DURATION_S,
    )


def _fake_diarization_result() -> DiarizationResult:
    return DiarizationResult(
        num_speakers=2,
        intervals=[
            SpeakerInterval(start=0.0, end=5.0, speaker="Speaker 0"),
            SpeakerInterval(start=5.5, end=10.0, speaker="Speaker 1"),
        ],
    )


async def _run_pipeline_with_fakes(
    manager: JobManager,
    job_id: str,
    media_path: Path,
    *,
    transcribe_impl=None,
    diarizer=None,
    enable_diarization: bool = False,
):
    """Drive run_pipeline end-to-end with all external dependencies faked."""
    fake_transcriber = MagicMock()
    fake_transcriber.name = "openai"
    fake_transcriber.transcribe.side_effect = (
        transcribe_impl or (lambda audio, **kw: _fake_transcription_result())
    )

    audio_processor = MagicMock()
    audio_processor.convert_to_whisper_wav.side_effect = lambda src, dst: dst
    metadata = MagicMock()
    metadata.duration = AUDIO_DURATION_S
    audio_processor.probe_media.return_value = metadata

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("jobs.AudioProcessor", return_value=audio_processor))
        stack.enter_context(patch("jobs.transcriber_factory.get_transcriber", return_value=fake_transcriber))
        if diarizer is not None:
            stack.enter_context(patch("diarization.diarizer_factory.get_diarizer", return_value=diarizer))
            stack.enter_context(patch("diarization.align_speakers_to_segments"))
        await manager.run_pipeline(
            job_id,
            media_path,
            whisper_engine="openai",
            whisper_model="whisper-large-v3-turbo",
            language="uk",
            enable_diarization=enable_diarization,
            ai_action="raw",
        )
    return fake_transcriber


def _drain_status_events(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        payload = queue.get_nowait()
        if payload["event"] == "status":
            events.append(payload["data"])
    return events


def _tmp_media(tmp_path: Path, name: str = "job_audio.wav") -> Path:
    media = tmp_path / name
    media.write_bytes(b"RIFF" + b"\x00" * 100)
    return media


# --- P6: queue waits become visible — "queued" (5) BEFORE the semaphore ----

@pytest.mark.asyncio
async def test_queued_status_emitted_before_semaphore_acquisition(tmp_path: Path):
    manager = JobManager()
    manager._semaphore = asyncio.Semaphore(0)  # saturated: pipeline must wait
    job = manager.create_job("queued.wav")
    queue = manager.subscribe(job.job_id)

    task = asyncio.create_task(
        manager.run_pipeline(job.job_id, _tmp_media(tmp_path), ai_action="raw")
    )
    try:
        event = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert event["event"] == "status"
        assert event["data"]["status"] == "queued"
        assert event["data"]["progress"] == 5
        assert event["data"]["message"]
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# --- P7 (jobs side): chunk progress mapped into the 35→50 band -------------

@pytest.mark.asyncio
async def test_chunk_progress_events_follow_35_to_50_band_formula(tmp_path: Path):
    manager = JobManager()
    job = manager.create_job("chunked.wav")
    queue = manager.subscribe(job.job_id)

    def fake_transcribe(audio, **kw):
        cb = kw.get("on_progress")
        assert callable(cb), "run_pipeline must pass an on_progress callback to transcribe()"
        cb(0.0, 0.0, "compressing")
        cb(60.0, 120.0, "chunk")
        cb(120.0, 120.0, "chunk")
        return _fake_transcription_result()

    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), transcribe_impl=fake_transcribe
    )
    assert job.status == "completed", f"pipeline must complete; got {job.status}: {job.error}"
    statuses = _drain_status_events(queue)
    transcribing = [s for s in statuses if s["status"] == "transcribing"]

    # Baseline marker first, then message-only "compressing" (no percent change)
    assert transcribing, "expected transcribing status events"
    assert transcribing[0]["progress"] == 35
    compressing = [s for s in transcribing if "ompress" in s["message"]]
    assert compressing, "expected a message-only 'compressing' status event"
    assert all(s["progress"] == 35 for s in compressing)

    # Chunk progress: exact band formula 35 + int(15*done/total)
    chunk_events = [s for s in transcribing if s["progress"] > 35]
    assert [s["progress"] for s in chunk_events] == [42, 50]
    assert all(35 <= s["progress"] <= 50 for s in chunk_events)


@pytest.mark.asyncio
async def test_chunk_progress_message_only_when_total_audio_unknown(tmp_path: Path):
    manager = JobManager()
    job = manager.create_job("unknown_total.wav")
    queue = manager.subscribe(job.job_id)

    def fake_transcribe(audio, **kw):
        cb = kw.get("on_progress")
        assert callable(cb), "run_pipeline must pass an on_progress callback to transcribe()"
        cb(30.0, 0.0, "chunk")  # total unknown → no percent may be invented
        return _fake_transcription_result()

    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), transcribe_impl=fake_transcribe
    )
    assert job.status == "completed", f"pipeline must complete; got {job.status}: {job.error}"
    statuses = _drain_status_events(queue)
    transcribing = [s for s in statuses if s["status"] == "transcribing"]
    assert transcribing
    assert all(s["progress"] == 35 for s in transcribing), "no percent may be invented when total is unknown"
    assert any(s["message"] for s in transcribing)


# --- P5: per-stage INFO logs with durations (observable contract, caplog) --

@pytest.mark.asyncio
async def test_run_pipeline_logs_per_stage_durations(caplog, tmp_path: Path, monkeypatch):
    caplog.set_level(logging.INFO, logger="jobs")

    # T7/P13: the diarize call moved to the process pool — stage logs are
    # unchanged, only the seam through which the result arrives.
    import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_diarize_worker", _pool_result_worker)

    diarizer = MagicMock()
    diarizer.is_available.return_value = True
    diarizer.models_ready.return_value = True

    manager = JobManager()
    job = manager.create_job("logged.wav")
    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), diarizer=diarizer, enable_diarization=True
    )

    msgs = [r.getMessage() for r in caplog.records if r.name == "jobs"]

    assert any(
        "started" in m and "engine=openai" in m and "model=whisper-large-v3-turbo" in m and "language=uk" in m
        for m in msgs
    ), "job start log with engine/model/language"
    assert any("audio converted" in m and f"duration={AUDIO_DURATION_S:.1f}" in m for m in msgs), "convert-done log with audio duration"
    assert any("transcription done" in m and "2 segments" in m and "RTF" in m for m in msgs), "transcribe-done log with segment count and RTF"
    assert any("diarization done" in m and "RTF" in m and "2 speakers" in m for m in msgs), "diarize-done log with RTF and num_speakers"
    assert any("completed in" in m for m in msgs), "total-duration log"


# --- P9: diarization heartbeat + start/end logs -----------------------------

@pytest.mark.asyncio
async def test_diarization_heartbeat_reports_elapsed_without_fake_percent(monkeypatch, tmp_path: Path):
    import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "DIARIZATION_HEARTBEAT_SECONDS", 0.05)
    # T7/P13: the blocking call now runs in the process pool — the heartbeat
    # contract (elapsed seconds, no fake percents) is unchanged around the pool await.
    monkeypatch.setattr(jobs_module, "_diarize_worker", _pool_sleeping_worker)

    manager = JobManager()
    job = manager.create_job("heartbeat.wav")
    queue = manager.subscribe(job.job_id)

    diarizer = MagicMock()
    diarizer.is_available.return_value = True
    diarizer.models_ready.return_value = True

    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), diarizer=diarizer, enable_diarization=True
    )
    statuses = _drain_status_events(queue)
    heartbeats = [s for s in statuses if "Diarizing" in s["message"] and "elapsed" in s["message"]]

    assert heartbeats, "expected heartbeat status events while diarization is pending"
    assert all(s["progress"] == 60 for s in heartbeats), "heartbeat must not invent fake percentages"
    elapsed_values = [int(re.search(r"(\d+)s elapsed", s["message"]).group(1)) for s in heartbeats]
    assert elapsed_values == sorted(elapsed_values)


# ---------------------------------------------------------------------------
# T7 — AH-5 (spec §3.4 P13/P14, DEC-9/DEC-10).
# P13: diarize() must run in a child process (ProcessPoolExecutor), the job
# result must flow back unchanged, model download stays in the parent, and the
# T6 heartbeat must keep covering the pool await.
# Pool workers are module-level so ProcessPoolExecutor can pickle them by
# reference into the child (spawn start method on macOS).
# ---------------------------------------------------------------------------


def _pool_pid_worker(audio, num_speakers, cluster_threshold, engine):
    """Runs in the pool child: smuggles the child pid back via num_speakers."""
    return DiarizationResult(num_speakers=os.getpid(), intervals=[])


def _pool_sleeping_worker(audio, num_speakers, cluster_threshold, engine):
    time.sleep(0.3)
    return _fake_diarization_result()


def _pool_result_worker(audio, num_speakers, cluster_threshold, engine):
    return _fake_diarization_result()


def _pool_record_threshold_worker(audio, num_speakers, cluster_threshold, engine, record_path):
    """Child-side recorder: persists the threshold the pipeline actually passed."""
    Path(record_path).write_text(repr(float(cluster_threshold)))
    return _fake_diarization_result()


@pytest.mark.asyncio
async def test_diarize_runs_in_child_process_and_result_flows_back_unchanged(tmp_path: Path, monkeypatch):
    import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "_diarize_worker", _pool_pid_worker)

    manager = JobManager()
    job = manager.create_job("pool.wav")

    diarizer = MagicMock()
    diarizer.is_available.return_value = True
    diarizer.models_ready.return_value = True

    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), diarizer=diarizer, enable_diarization=True
    )

    assert job.status == "completed", f"pipeline must complete; got {job.status}: {job.error}"
    assert job.result is not None, "job result must be present after pool diarization"
    child_pid = job.result["num_speakers"]
    assert child_pid != os.getpid(), "diarize() must execute in a child process, not in the parent"
    assert child_pid > 1, "worker pid must be a real separate process"


@pytest.mark.asyncio
async def test_model_download_stays_in_parent_before_pool_submission(tmp_path: Path, monkeypatch):
    import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "_diarize_worker", _pool_pid_worker)

    manager = JobManager()
    job = manager.create_job("download.wav")
    queue = manager.subscribe(job.job_id)

    diarizer = MagicMock()
    diarizer.is_available.return_value = True
    diarizer.models_ready.return_value = False

    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), diarizer=diarizer, enable_diarization=True
    )

    assert job.status == "completed", f"pipeline must complete; got {job.status}: {job.error}"
    assert diarizer.ensure_models.called, (
        "models must be downloaded in the parent process before the pool submission"
    )
    statuses = _drain_status_events(queue)
    assert any("Downloading speaker diarization models" in s["message"] for s in statuses), (
        "download status event must still be emitted"
    )


@pytest.mark.asyncio
async def test_diarization_heartbeat_covers_process_pool_await(tmp_path: Path, monkeypatch):
    """T6 integration note: the heartbeat must keep working around the P13 pool await."""
    import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "DIARIZATION_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(jobs_module, "_diarize_worker", _pool_sleeping_worker)

    manager = JobManager()
    job = manager.create_job("heartbeat_pool.wav")
    queue = manager.subscribe(job.job_id)

    diarizer = MagicMock()
    diarizer.is_available.return_value = True
    diarizer.models_ready.return_value = True

    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), diarizer=diarizer, enable_diarization=True
    )
    assert job.status == "completed", f"pipeline must complete; got {job.status}: {job.error}"

    statuses = _drain_status_events(queue)
    heartbeats = [s for s in statuses if "Diarizing" in s["message"] and "elapsed" in s["message"]]
    assert heartbeats, "expected heartbeat status events while the pool worker is pending"
    assert all(s["progress"] == 60 for s in heartbeats), "heartbeat must not invent fake percentages"
    elapsed_values = [int(re.search(r"(\d+)s elapsed", s["message"]).group(1)) for s in heartbeats]
    assert elapsed_values == sorted(elapsed_values)


@pytest.mark.asyncio
async def test_run_pipeline_cluster_threshold_default_resolves_from_settings(tmp_path: Path, monkeypatch):
    import config as config_module
    import jobs as jobs_module

    monkeypatch.setattr(config_module.settings, "diarization_threshold", 0.42)
    record = tmp_path / "threshold.txt"
    monkeypatch.setattr(
        jobs_module,
        "_diarize_worker",
        functools.partial(_pool_record_threshold_worker, record_path=str(record)),
    )

    manager = JobManager()
    job = manager.create_job("threshold.wav")

    diarizer = MagicMock()
    diarizer.is_available.return_value = True
    diarizer.models_ready.return_value = True

    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), diarizer=diarizer, enable_diarization=True
    )
    assert job.status == "completed", f"pipeline must complete; got {job.status}: {job.error}"

    passed = float(record.read_text())
    assert passed == 0.42, (
        "run_pipeline's cluster_threshold default must resolve from settings.diarization_threshold"
    )


def test_sherpa_diarizer_logs_start_end_with_duration_and_rtf(caplog):
    from diarization.sherpa_diarizer import SherpaDiarizer

    caplog.set_level(logging.INFO, logger="diarization.sherpa_diarizer")

    fake_sherpa = MagicMock()
    processed = fake_sherpa.OfflineSpeakerDiarization.return_value.process.return_value
    processed.sort_by_start_time.return_value = []
    processed.num_speakers = 0

    samples = np.zeros(32000, dtype=np.float32)  # 2.0 s of audio
    with (
        patch("diarization.sherpa_diarizer.sherpa_onnx", fake_sherpa),
        patch.object(SherpaDiarizer, "models_ready", return_value=True),
        patch.object(SherpaDiarizer, "_load_audio_samples", return_value=samples),
    ):
        result = SherpaDiarizer().diarize(samples)

    assert result.num_speakers == 0
    msgs = [r.getMessage() for r in caplog.records if r.name == "diarization.sherpa_diarizer"]
    assert any("started" in m.lower() and "2.0" in m for m in msgs), "diarization start log with audio duration"
    assert any("done" in m.lower() and "RTF" in m for m in msgs), "diarization end log with duration and RTF"
