"""Unit and integration tests for speaker diarization and speaker-aware features."""

import contextlib
import io
import json
import os
import threading
import wave
from unittest.mock import MagicMock, patch

import httpx
import pytest
import numpy as np
from diarization.base import BaseDiarizer, SpeakerInterval, DiarizationResult
from diarization.factory import diarizer_factory
from diarization.alignment import compute_overlap, align_speakers_to_segments
from transcribers.base import Segment, Word, TranscriptionResult
from jobs import job_manager


def test_diarization_data_models():
    interval1 = SpeakerInterval(start=0.0, end=2.5, speaker="Speaker 0", confidence=0.92)
    interval2 = SpeakerInterval(start=2.8, end=5.0, speaker="Speaker 1", confidence=0.88)
    interval3 = SpeakerInterval(start=5.2, end=7.0, speaker="Speaker 0", confidence=0.95)

    result = DiarizationResult(
        num_speakers=2,
        intervals=[interval1, interval2, interval3]
    )

    assert result.num_speakers == 2
    assert result.speaker_names == ["Speaker 0", "Speaker 1"]
    assert len(result.intervals) == 3


def test_compute_overlap():
    assert compute_overlap(0.0, 5.0, 2.0, 4.0) == 2.0
    assert compute_overlap(0.0, 2.0, 3.0, 5.0) == 0.0
    assert compute_overlap(1.0, 4.0, 2.0, 6.0) == 2.0
    assert compute_overlap(5.0, 10.0, 4.0, 6.0) == 1.0


def test_align_speakers_to_segments_without_words():
    segments = [
        Segment(id=0, start=0.0, end=3.0, text="Hello there."),
        Segment(id=1, start=3.5, end=6.0, text="General Kenobi!"),
    ]
    diarization = DiarizationResult(
        num_speakers=2,
        intervals=[
            SpeakerInterval(start=0.0, end=2.8, speaker="Speaker 0"),
            SpeakerInterval(start=3.2, end=6.2, speaker="Speaker 1"),
        ]
    )

    aligned = align_speakers_to_segments(segments, diarization)
    assert aligned[0].speaker == "Speaker 0"
    assert aligned[1].speaker == "Speaker 1"


def test_align_speakers_to_segments_with_words():
    words1 = [
        Word(word="Good", start=0.0, end=0.5),
        Word(word="morning", start=0.6, end=1.2),
        Word(word="everyone", start=1.3, end=2.0),
    ]
    words2 = [
        Word(word="Welcome", start=2.5, end=3.0),
        Word(word="back", start=3.1, end=3.8),
    ]
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Good morning everyone", words=words1),
        Segment(id=1, start=2.5, end=3.8, text="Welcome back", words=words2),
    ]
    diarization = DiarizationResult(
        num_speakers=2,
        intervals=[
            SpeakerInterval(start=0.0, end=2.2, speaker="Speaker 0"),
            SpeakerInterval(start=2.4, end=4.0, speaker="Speaker 1"),
        ]
    )

    aligned = align_speakers_to_segments(segments, diarization)
    assert aligned[0].speaker == "Speaker 0"
    assert aligned[0].words[0].speaker == "Speaker 0"
    assert aligned[1].speaker == "Speaker 1"
    assert aligned[1].words[0].speaker == "Speaker 1"


def test_align_speakers_empty_intervals_fallback():
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Single speaker recording."),
    ]
    diarization = DiarizationResult(num_speakers=0, intervals=[])

    aligned = align_speakers_to_segments(segments, diarization, default_speaker="Speaker 0")
    assert aligned[0].speaker == "Speaker 0"


def test_transcription_result_speaker_formatting():
    seg1 = Segment(id=0, start=0.0, end=2.0, text="First utterance", speaker="Speaker 0")
    seg2 = Segment(id=1, start=2.2, end=4.0, text="Second utterance", speaker="Speaker 0")
    seg3 = Segment(id=2, start=4.5, end=7.0, text="Reply from second speaker", speaker="Speaker 1")

    res = TranscriptionResult(
        text="First utterance Second utterance Reply from second speaker",
        segments=[seg1, seg2, seg3],
        duration=7.0,
    )

    # Test speaker dialogue turns
    turns = res.get_speaker_turns()
    assert len(turns) == 2
    assert turns[0]["speaker"] == "Speaker 0"
    assert turns[0]["text"] == "First utterance Second utterance"
    assert turns[0]["start"] == 0.0
    assert turns[0]["end"] == 4.0
    assert turns[1]["speaker"] == "Speaker 1"
    assert turns[1]["text"] == "Reply from second speaker"

    # Test TXT formatting
    txt = res.to_txt()
    assert "[Speaker 0]:" in txt
    assert "[Speaker 1]:" in txt

    # Test SRT formatting
    srt = res.to_srt()
    assert "Speaker 0: First utterance" in srt
    assert "Speaker 1: Reply from second speaker" in srt

    # Test WebVTT formatting
    vtt = res.to_vtt()
    assert "<v Speaker 0>First utterance</v>" in vtt
    assert "<v Speaker 1>Reply from second speaker</v>" in vtt

    # Test ASS formatting
    ass = res.to_ass()
    assert "Karaoke,Speaker 0" in ass or "Default,Speaker 0" in ass
    assert "Karaoke,Speaker 1" in ass or "Default,Speaker 1" in ass


def test_diarizer_factory():
    # AH-18 (DEC-16/17): the registry keeps its shape, but `remote` is the only
    # engine; get_diarizer()'s default is "remote"; unknown names raise KeyError.
    engines = diarizer_factory.list_engines()
    assert [e["name"] for e in engines] == ["remote"], (
        "remote must be the only registered engine after AH-18"
    )

    from diarization.remote_diarizer import RemoteDiarizer

    diarizer = diarizer_factory.get_diarizer("remote")
    assert isinstance(diarizer, RemoteDiarizer)

    default_diarizer = diarizer_factory.get_diarizer()
    assert isinstance(default_diarizer, RemoteDiarizer), (
        "get_diarizer() default must resolve the remote engine (DEC-17)"
    )

    with pytest.raises(KeyError):
        diarizer_factory.get_diarizer("no-such-engine")


def test_job_manager_rename_speaker():
    job = job_manager.create_job("meeting.wav")
    job.result = {
        "text": "[Speaker 0]: Hello\n\n[Speaker 1]: World",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello", "speaker": "Speaker 0", "words": [{"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "Speaker 0"}]},
            {"id": 1, "start": 1.5, "end": 2.5, "text": "World", "speaker": "Speaker 1", "words": [{"word": "World", "start": 1.5, "end": 2.5, "speaker": "Speaker 1"}]},
        ],
        "language": "en",
        "duration": 2.5,
    }

    updated_result = job_manager.rename_speaker(job.job_id, "Speaker 0", "Alice")
    assert updated_result is not None
    assert updated_result["segments"][0]["speaker"] == "Alice"
    assert updated_result["segments"][0]["words"][0]["speaker"] == "Alice"
    assert updated_result["segments"][1]["speaker"] == "Speaker 1"
    assert "Alice: Hello" in updated_result["srt"]
    assert "<v Alice>Hello</v>" in updated_result["vtt"]
    assert "[Alice]:" in updated_result["text"]
    assert updated_result["speaker_turns"][0]["speaker"] == "Alice"


# ---------------------------------------------------------------------------
# T10 — AH-11b remote engine + de-blip filter (spec §3.5 R1/R1'/R2/R2'/R3/
# R4/R5/R7', DEC-14/DEC-15). Mocked httpx (MockTransport) — no live sidecar.
# Behavior assertions only; the filter tests are the canonical implementation
# of the rule RG proved offline (test-strategy-ah11b.md).
# ---------------------------------------------------------------------------

API_URL = "http://sidecar.test:8000"


def _payload(intervals, num_speakers):
    return {"num_speakers": num_speakers, "intervals": intervals}


def _iv(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def _mock_transport(payload, captured=None):
    """httpx.MockTransport answering POST /diarize; optionally records requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _remote(monkeypatch, payload, captured=None, floor=None, api_url=API_URL):
    import config as config_module

    monkeypatch.setattr(config_module.settings, "diarization_api_url", api_url)
    if floor is not None:
        monkeypatch.setattr(config_module.settings, "diarization_min_speaker_duration", floor)

    from diarization.remote_diarizer import RemoteDiarizer

    return RemoteDiarizer(transport=_mock_transport(payload, captured))


# --- R7'(a): response → DiarizationResult shape (filter off: raw passthrough) ---

def test_remote_diarizer_maps_response_to_diarization_result(monkeypatch):
    payload = _payload(
        [_iv(0.0, 6.0, "spk0"), _iv(6.5, 12.0, "spk1")],
        2,
    )
    diarizer = _remote(monkeypatch, payload, floor=0.0)
    result = diarizer.diarize(np.zeros(1600, dtype=np.float32))

    assert isinstance(result, DiarizationResult)
    assert result.num_speakers == 2
    assert [(i.start, i.end, i.speaker) for i in result.intervals] == [
        (0.0, 6.0, "spk0"),
        (6.5, 12.0, "spk1"),
    ]


def test_remote_diarizer_posts_to_api_url_diarize_endpoint(monkeypatch):
    captured = []
    diarizer = _remote(monkeypatch, _payload([], 0), captured, floor=0.0)
    diarizer.diarize(np.zeros(1600, dtype=np.float32))

    assert len(captured) == 1
    assert str(captured[0].url) == f"{API_URL}/diarize"
    assert captured[0].headers["content-type"].startswith("multipart/")


# --- R7'(b): is_available() gated on settings.diarization_api_url, no probe ---

def test_remote_diarizer_is_available_false_without_api_url(monkeypatch):
    import config as config_module

    monkeypatch.setattr(config_module.settings, "diarization_api_url", "")
    from diarization.remote_diarizer import RemoteDiarizer

    assert RemoteDiarizer().is_available() is False


def test_remote_diarizer_is_available_true_with_api_url(monkeypatch):
    import config as config_module

    monkeypatch.setattr(config_module.settings, "diarization_api_url", API_URL)
    from diarization.remote_diarizer import RemoteDiarizer

    # No health probe: availability is config presence alone.
    assert RemoteDiarizer().is_available() is True


# --- R7'(c): float32 ndarray → valid WAV multipart (stdlib wave round-trip) ---

def _multipart_file_bytes(request: httpx.Request, field_name: str = "file") -> bytes:
    body = request.read()
    boundary = request.headers["content-type"].split("boundary=")[1].strip('"').encode()
    for part in body.split(b"--" + boundary):
        if f'name="{field_name}"'.encode() in part:
            _, _, data = part.partition(b"\r\n\r\n")
            return data[: -2] if data.endswith(b"\r\n") else data
    raise AssertionError(f"multipart body has no '{field_name}' field")


def test_remote_diarizer_uploads_valid_wav_for_ndarray_input(monkeypatch):
    captured = []
    diarizer = _remote(monkeypatch, _payload([], 0), captured, floor=0.0)

    # Deterministic ramp signal, 0.1 s @ 16 kHz float32
    samples = (np.arange(1600, dtype=np.float32) / 1600.0) - 0.5
    diarizer.diarize(samples)

    wav_bytes = _multipart_file_bytes(captured[0])
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # int16
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 1600
        decoded = np.frombuffer(wf.readframes(1600), dtype=np.int16).astype(np.float64)

    expected = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).astype(np.float64)
    max_err = np.max(np.abs(decoded - expected))
    assert max_err <= 1.0, f"int16 round-trip error too large: {max_err}"


def test_remote_diarizer_uploads_file_bytes_for_path_input(monkeypatch, tmp_path):
    captured = []
    diarizer = _remote(monkeypatch, _payload([], 0), captured, floor=0.0)

    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(800, dtype=np.int16).tobytes())
    original = wav_path.read_bytes()

    diarizer.diarize(wav_path)

    uploaded = _multipart_file_bytes(captured[0])
    assert uploaded == original, "Path input must be uploaded as the file's own bytes"


# --- R7'(f): de-blip filter (R2′) — canonical implementation of the RG rule ---

def test_filter_drops_blip_speaker_below_floor(monkeypatch):
    # Mirrors the RG case: spk2 = 1 interval, 0.51 s total (RG: t=340.42).
    payload = _payload(
        [
            _iv(0.0, 10.0, "spk0"),
            _iv(12.0, 16.0, "spk1"),
            _iv(20.0, 25.0, "spk0"),
            _iv(340.42, 340.93, "spk2"),
        ],
        3,
    )
    diarizer = _remote(monkeypatch, payload, floor=2.0)
    result = diarizer.diarize(np.zeros(1600, dtype=np.float32))

    assert result.num_speakers == 2
    assert all(i.speaker != "spk2" for i in result.intervals), "blip speaker must be dropped"
    assert len(result.intervals) == 3


def test_filter_preserves_short_turns_of_kept_speakers(monkeypatch):
    # Per-SPEAKER, not per-interval: a 0.3 s turn of a kept speaker survives.
    payload = _payload(
        [
            _iv(0.0, 8.0, "spkA"),
            _iv(10.0, 14.0, "spkB"),
            _iv(30.0, 30.3, "spkA"),
        ],
        2,
    )
    diarizer = _remote(monkeypatch, payload, floor=2.0)
    result = diarizer.diarize(np.zeros(1600, dtype=np.float32))

    assert result.num_speakers == 2
    assert len(result.intervals) == 3, "short turns of KEPT speakers must be preserved"
    # Survivors are re-labeled in arrival order: spkA → spk0, spkB → spk1.
    assert [i.speaker for i in result.intervals] == ["spk0", "spk1", "spk0"]
    short = [i for i in result.intervals if i.end - i.start < 1.0]
    assert len(short) == 1 and short[0].speaker == "spk0", (
        "the kept speaker's short turn must survive the filter"
    )


def test_filter_relabels_survivors_in_arrival_order_and_recounts(monkeypatch):
    # Arrival order of survivors: spk5 first, then spk9 (spk1 dropped: 0.4 s).
    payload = _payload(
        [
            _iv(0.0, 6.0, "spk5"),
            _iv(10.0, 10.4, "spk1"),
            _iv(20.0, 25.0, "spk9"),
        ],
        3,
    )
    diarizer = _remote(monkeypatch, payload, floor=2.0)
    result = diarizer.diarize(np.zeros(1600, dtype=np.float32))

    assert result.num_speakers == 2
    assert [i.speaker for i in result.intervals] == ["spk0", "spk1"], (
        "survivors must be re-labeled in arrival order"
    )


def test_filter_disabled_when_floor_is_zero(monkeypatch):
    payload = _payload(
        [
            _iv(0.0, 10.0, "spk0"),
            _iv(12.0, 16.0, "spk1"),
            _iv(340.42, 340.93, "spk2"),
        ],
        3,
    )
    diarizer = _remote(monkeypatch, payload, floor=0.0)
    result = diarizer.diarize(np.zeros(1600, dtype=np.float32))

    assert result.num_speakers == 3
    assert [i.speaker for i in result.intervals] == ["spk0", "spk1", "spk2"], (
        "floor 0 must disable the filter entirely (raw labels preserved)"
    )


def test_filter_all_speakers_below_floor_yields_empty_result(monkeypatch):
    payload = _payload(
        [_iv(0.0, 0.5, "spk0"), _iv(1.0, 1.4, "spk1")],
        2,
    )
    diarizer = _remote(monkeypatch, payload, floor=2.0)
    result = diarizer.diarize(np.zeros(1600, dtype=np.float32))

    assert result.num_speakers == 0
    assert result.intervals == []


# --- R7'(g): settings round-trip — config.json dict + env (P10 pattern) ---

def test_diarization_engine_settings_defaults(monkeypatch):
    for var in (
        "DIARIZATION_ENGINE",
        "DIARIZATION_API_URL",
        "DIARIZATION_REQUEST_TIMEOUT",
        "DIARIZATION_MIN_SPEAKER_DURATION",
    ):
        monkeypatch.delenv(var, raising=False)
    from config import Settings

    s = Settings()
    assert s.diarization_engine == "remote"  # single-engine default (DEC-17, AH-18)
    assert s.diarization_api_url == ""
    assert s.diarization_request_timeout == 300
    assert s.diarization_min_speaker_duration == 2.0


def test_diarization_engine_settings_env_overrides(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ENGINE", "remote")
    monkeypatch.setenv("DIARIZATION_API_URL", API_URL)
    monkeypatch.setenv("DIARIZATION_REQUEST_TIMEOUT", "42")
    monkeypatch.setenv("DIARIZATION_MIN_SPEAKER_DURATION", "0")
    from config import Settings

    s = Settings()
    assert s.diarization_engine == "remote"
    assert s.diarization_api_url == API_URL
    assert s.diarization_request_timeout == 42
    assert s.diarization_min_speaker_duration == 0.0


def test_diarization_engine_settings_config_json_overrides(monkeypatch):
    import config as config_module

    monkeypatch.setattr(
        config_module,
        "_diarization_cfg",
        {
            "engine": "remote",
            "api_url": API_URL,
            "request_timeout": 77,
            "min_speaker_duration": 1.5,
        },
    )
    s = config_module.Settings()
    assert s.diarization_engine == "remote"
    assert s.diarization_api_url == API_URL
    assert s.diarization_request_timeout == 77
    assert s.diarization_min_speaker_duration == 1.5


# --- R5: factory registration ---

def test_remote_engine_registered_in_factory(monkeypatch):
    import config as config_module

    monkeypatch.setattr(config_module.settings, "diarization_api_url", "")
    engines = diarizer_factory.list_engines()
    entry = next((e for e in engines if e["name"] == "remote"), None)
    assert entry is not None, "remote engine must be registered in the factory"
    assert entry["available"] is False  # no API URL configured in this test env


# --- R4: engine resolution at the call sites (jobs.py run_pipeline, main.py sync path) ---


def _fake_transcription_result():
    return TranscriptionResult(
        text="hello world",
        segments=[Segment(id=0, start=0.0, end=5.0, text="seg 0")],
        language="en",
        duration=120.0,
    )


def _fake_diarization_result():
    return DiarizationResult(
        num_speakers=2,
        intervals=[
            SpeakerInterval(start=0.0, end=5.0, speaker="spk0"),
            SpeakerInterval(start=5.5, end=10.0, speaker="spk1"),
        ],
    )


async def _run_pipeline_with_fakes(manager, job_id, media_path, *, diarizer, get_diarizer_patch=None):
    """Drive run_pipeline with all external dependencies faked (test_jobs pattern)."""
    from jobs import JobManager

    fake_transcriber = MagicMock()
    fake_transcriber.name = "openai"
    fake_transcriber.transcribe.side_effect = lambda audio, **kw: _fake_transcription_result()

    audio_processor = MagicMock()
    audio_processor.convert_to_whisper_wav.side_effect = lambda src, dst: dst
    metadata = MagicMock()
    metadata.duration = 120.0
    audio_processor.probe_media.return_value = metadata

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("jobs.AudioProcessor", return_value=audio_processor))
        stack.enter_context(patch("jobs.transcriber_factory.get_transcriber", return_value=fake_transcriber))
        if get_diarizer_patch is not None:
            stack.enter_context(patch("diarization.diarizer_factory.get_diarizer", get_diarizer_patch))
        else:
            stack.enter_context(patch("diarization.diarizer_factory.get_diarizer", return_value=diarizer))
        stack.enter_context(patch("diarization.align_speakers_to_segments"))
        await manager.run_pipeline(
            job_id,
            media_path,
            whisper_engine="openai",
            whisper_model="whisper-large-v3-turbo",
            language="uk",
            enable_diarization=True,
            ai_action="raw",
        )


def _tmp_media(tmp_path, name="job_audio.wav"):
    media = tmp_path / name
    media.write_bytes(b"RIFF" + b"\x00" * 100)
    return media


def _available_diarizer_mock():
    diarizer = MagicMock()
    diarizer.is_available.return_value = True
    diarizer.models_ready.return_value = True
    return diarizer


async def test_run_pipeline_resolves_diarizer_engine_from_settings(tmp_path, monkeypatch):
    """jobs.py:281 — factory must receive settings.diarization_engine, not a constant."""
    import config as config_module
    from jobs import JobManager

    monkeypatch.setattr(config_module.settings, "diarization_engine", "test-engine")

    diarizer = _available_diarizer_mock()
    diarizer.use_process_pool = False
    diarizer.diarize.return_value = _fake_diarization_result()
    get_spy = MagicMock(return_value=diarizer)

    manager = JobManager()
    job = manager.create_job("engine.wav")
    await _run_pipeline_with_fakes(
        manager, job.job_id, _tmp_media(tmp_path), diarizer=diarizer, get_diarizer_patch=get_spy
    )

    assert job.status == "completed", f"pipeline must complete; got {job.status}: {job.error}"
    requested = [c.args[0] for c in get_spy.call_args_list if c.args]
    assert "test-engine" in requested, (
        "run_pipeline must resolve the engine from settings.diarization_engine"
    )


async def test_sync_transcribe_endpoint_resolves_engine_from_settings(tmp_path, monkeypatch):
    """main.py:540 — the synchronous /api/transcribe path must resolve the engine from settings."""
    import config as config_module

    monkeypatch.setattr(config_module.settings, "diarization_engine", "test-engine")

    diarizer = _available_diarizer_mock()
    diarizer.diarize.return_value = _fake_diarization_result()
    get_spy = MagicMock(return_value=diarizer)

    fake_transcriber = MagicMock()
    fake_transcriber.name = "openai"
    fake_transcriber.transcribe.return_value = _fake_transcription_result()

    audio_processor = MagicMock()
    audio_processor.convert_to_whisper_wav.side_effect = lambda src, dst: dst
    metadata = MagicMock()
    metadata.duration = 1.0
    audio_processor.probe_media.return_value = metadata

    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("main.transcriber_factory.get_transcriber", return_value=fake_transcriber))
        stack.enter_context(patch("main.audio_processor", audio_processor))
        stack.enter_context(patch("diarization.diarizer_factory.get_diarizer", get_spy))

        response = client.post(
            "/api/transcribe",
            files={"file": ("t.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data={"whisper_engine": "openai", "whisper_model": "m", "enable_diarization": "true"},
        )

    assert response.status_code == 200, f"endpoint must serve the job; got {response.status_code}"
    requested = [c.args[0] for c in get_spy.call_args_list if c.args]
    assert "test-engine" in requested, (
        "/api/transcribe must resolve the engine from settings.diarization_engine"
    )


# --- AH-18 (DEC-18): single executor path — diarization runs in the default thread executor ---

async def test_diarization_runs_in_default_thread_executor(tmp_path, monkeypatch):
    """AH-18: with the process-pool machinery removed there is ONE path — the
    diarize() call runs in a worker THREAD of the SAME process (not the main
    thread, never a child process)."""
    from jobs import JobManager

    record = tmp_path / "executor.json"
    diarizer = _available_diarizer_mock()

    def _record_and_result(audio, *args, **kwargs):
        rec = {
            "pid": os.getpid(),
            "is_main_thread": threading.current_thread() is threading.main_thread(),
        }
        record.write_text(json.dumps(rec))
        return _fake_diarization_result()

    diarizer.diarize.side_effect = _record_and_result

    manager = JobManager()
    job = manager.create_job("thread.wav")
    await _run_pipeline_with_fakes(manager, job.job_id, _tmp_media(tmp_path), diarizer=diarizer)

    assert job.status == "completed", f"pipeline must complete; got {job.status}: {job.error}"
    rec = json.loads(record.read_text())
    assert rec["pid"] == os.getpid(), "thread executor must run in the same process"
    assert rec["is_main_thread"] is False, "diarize() must run in a worker thread, not the event loop"
