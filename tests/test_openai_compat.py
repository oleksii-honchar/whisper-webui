"""Unit tests for OpenAI-compatible STT & LLM universal adapters and settings."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import Settings, get_masked_settings, mask_secret, save_config, settings
from llm import llm_registry
from llm.openai_compat import OpenAICompatibleLLMProvider
from main import app
from transcribers.base import Segment
from transcribers.factory import TranscriberFactory, transcriber_factory
from transcribers.openai_compat import OpenAICompatibleTranscriber

client = TestClient(app)


def test_mask_secret():
    assert mask_secret(None) == ""
    assert mask_secret("") == ""
    assert mask_secret("short") == "••••••••"
    assert mask_secret("gsk_1234567890abcdef") == "gsk_••••cdef"
    assert mask_secret("sk-or-v1-abcdef123456789") == "sk-o••••6789"


def test_openai_transcriber_initialization():
    fake_key = "gsk_test123456789"
    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1/",
        api_key_getter=lambda: fake_key,
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo", "whisper-large-v3"],
        extra_headers={"X-Custom": "TestVal"},
    )

    assert transcriber.name == "groq"
    assert transcriber.display_name == "Groq Cloud Whisper"
    assert transcriber.get_base_url() == "https://api.groq.com/openai/v1"
    assert transcriber.get_api_key() == fake_key
    assert transcriber.is_available() is True
    assert transcriber.get_extra_headers() == {"X-Custom": "TestVal"}


def test_openai_transcriber_unconfigured_availability():
    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1",
        api_key_getter=lambda: None,
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )
    assert transcriber.is_available() is False
    with pytest.raises(RuntimeError, match="API key is not configured"):
        transcriber.transcribe("dummy_file.wav")


def test_openai_transcriber_parse_verbose_json():
    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1",
        api_key_getter="fake_key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )

    verbose_response = {
        "text": "Hello world and welcome to speech recognition.",
        "language": "english",
        "duration": 5.2,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": " Hello world",
                "avg_logprob": -0.105,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.99},
                    {"word": "world", "start": 1.1, "end": 2.5, "probability": 0.98},
                ],
            },
            {
                "id": 1,
                "start": 2.6,
                "end": 5.2,
                "text": " and welcome to speech recognition.",
                "avg_logprob": -0.05,
                "words": [
                    {"word": "and", "start": 2.6, "end": 3.0, "probability": 0.95},
                    {"word": "welcome", "start": 3.1, "end": 5.2, "probability": 0.97},
                ],
            },
        ],
    }

    emitted_segments: list[Segment] = []
    segs, text, lang, dur = transcriber._parse_verbose_json(
        data=verbose_response,
        time_offset=10.0,
        start_segment_id=5,
        on_segment=lambda s: emitted_segments.append(s),
    )

    assert len(segs) == 2
    assert len(emitted_segments) == 2
    assert segs[0].id == 5
    assert segs[0].start == 10.0
    assert segs[0].end == 12.5
    assert segs[0].text == "Hello world"
    assert segs[0].confidence is not None
    assert len(segs[0].words) == 2
    assert segs[0].words[0].word == "Hello"
    assert segs[0].words[0].start == 10.0
    assert segs[0].words[0].end == 11.0

    assert segs[1].id == 6
    assert segs[1].start == 12.6
    assert segs[1].end == 15.2
    assert lang == "english"
    assert dur == 5.2


def test_openai_transcriber_transcribe_with_mock(tmp_path: Path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1",
        api_key_getter="gsk_valid_key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Transcribed test content.",
        "language": "en",
        "duration": 3.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "Transcribed test content.",
                "words": [
                    {"word": "Transcribed", "start": 0.0, "end": 1.0},
                    {"word": "test", "start": 1.1, "end": 1.8},
                    {"word": "content.", "start": 1.9, "end": 3.0},
                ],
            }
        ],
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = transcriber.transcribe(audio_file, model_name="whisper-large-v3-turbo")

        assert result.text == "Transcribed test content."
        assert len(result.segments) == 1
        assert len(result.segments[0].words) == 3
        assert result.duration == 3.0
        assert result.language == "en"

        # Check call arguments
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert call_url == "https://api.groq.com/openai/v1/audio/transcriptions"
        call_headers = mock_post.call_args[1]["headers"]
        assert call_headers["Authorization"] == "Bearer gsk_valid_key"


def test_openai_transcriber_numpy_input():
    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1",
        api_key_getter="gsk_valid_key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )

    dummy_pcm = np.zeros(16000, dtype=np.float32)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "NumPy test passed.",
        "language": "en",
        "duration": 1.0,
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "NumPy test passed.", "words": []}],
    }

    with patch("httpx.Client.post", return_value=mock_response):
        result = transcriber.transcribe(dummy_pcm)
        assert result.text == "NumPy test passed."


def test_openai_transcriber_fallback_when_granularities_unsupported(tmp_path: Path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    transcriber = OpenAICompatibleTranscriber(
        name="custom",
        display_name="Custom STT",
        base_url_getter="https://api.custom.ai/v1",
        api_key_getter="sk_custom",
        default_model="whisper-1",
        supported_models=["whisper-1"],
    )

    err_response = MagicMock()
    err_response.status_code = 400
    err_response.text = "Unrecognized parameter: timestamp_granularities"

    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {
        "text": "Fallback succeeded.",
        "language": "en",
        "duration": 2.0,
        "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "Fallback succeeded.", "words": []}],
    }

    with patch("httpx.Client.post", side_effect=[err_response, ok_response]) as mock_post:
        result = transcriber.transcribe(audio_file)
        assert result.text == "Fallback succeeded."
        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_openai_llm_provider_streaming_mock():
    provider = OpenAICompatibleLLMProvider(
        provider_id="groq",
        display_name="Groq Cloud",
        api_key_getter="gsk_test",
        base_url_getter="https://api.groq.com/openai/v1",
        default_model_getter="llama-3.3-70b-versatile",
    )

    assert provider.is_configured() is True
    assert provider.get_default_model() == "llama-3.3-70b-versatile"

    sse_lines = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line.decode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp
        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient.stream", return_value=MockStreamContext()):
        collected = []
        async for token in provider.generate_stream("Say hello"):
            collected.append(token)
        assert "".join(collected) == "Hello world"


@pytest.mark.asyncio
async def test_openai_llm_provider_list_models():
    provider = OpenAICompatibleLLMProvider(
        provider_id="openrouter",
        display_name="OpenRouter",
        api_key_getter="sk-or-test",
        base_url_getter="https://openrouter.ai/api/v1",
        default_model_getter="meta-llama/llama-3.3-70b-instruct",
        curated_models=["meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-r1"],
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "deepseek/deepseek-r1"},
            {"id": "meta-llama/llama-3.3-70b-instruct"},
            {"id": "mistralai/mistral-large"},
        ]
    }

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        models = await provider.list_models()
        assert models[0] == "meta-llama/llama-3.3-70b-instruct"
        assert models[1] == "deepseek/deepseek-r1"
        assert "mistralai/mistral-large" in models


def test_transcriber_factory_engines_and_models():
    engines = transcriber_factory.list_engines()
    engine_ids = [e["id"] for e in engines]
    assert "groq" in engine_ids
    assert "openrouter" in engine_ids
    assert "openai" in engine_ids

    groq_models = transcriber_factory.get_models_for_engine("groq")
    assert "whisper-large-v3-turbo" in groq_models
    assert "whisper-large-v3" in groq_models

    or_models = transcriber_factory.get_models_for_engine("openrouter")
    assert "openai/whisper-large-v3" in or_models


def test_api_whisper_models_endpoint():
    resp = client.get("/api/whisper/models?engine=groq")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "groq"
    assert "whisper-large-v3-turbo" in data["models"]


def test_settings_endpoints():
    # 1. GET /api/settings
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    settings_data = resp.json()
    assert "groq" in settings_data
    assert "openrouter" in settings_data

    # 2. POST /api/settings
    update_payload = {
        "groq": {"default_model": "llama-3.3-70b-versatile"},
        "openrouter": {"default_model": "meta-llama/llama-3.3-70b-instruct"},
    }
    resp_post = client.post("/api/settings", json=update_payload)
    assert resp_post.status_code == 200
    assert resp_post.json()["status"] == "ok"


def test_settings_test_endpoint():
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        resp = client.post(
            "/api/settings/test",
            json={"provider": "groq", "api_key": "gsk_dummy_test_key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["online"] is True
        assert "latency_ms" in data


@pytest.mark.asyncio
async def test_openai_llm_provider_model_aliases():
    provider = OpenAICompatibleLLMProvider(
        provider_id="custom",
        display_name="Custom Endpoint",
        api_key_getter="sk-test",
        base_url_getter="https://api.custom.com/v1",
        default_model_getter="gpt-4o-mini",
        model_aliases={"legacy-model": "modern-model-v2", "gemini-2.5-flash": "gemini-3.8-flash"},
    )

    async def mock_aiter_lines():
        yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp
        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient.stream", return_value=MockStreamContext()) as mock_stream:
        tokens = [t async for t in provider.generate_stream("test", model="legacy-model")]
        assert "".join(tokens) == "ok"
        mock_stream.assert_called_once()
        call_json = mock_stream.call_args[1]["json"]
        # Verify alias was resolved
        assert call_json["model"] == "modern-model-v2"


@pytest.mark.asyncio
async def test_openai_llm_provider_exclude_keywords():
    provider = OpenAICompatibleLLMProvider(
        provider_id="custom",
        display_name="Custom Endpoint",
        api_key_getter="sk-test",
        base_url_getter="https://api.custom.com/v1",
        curated_models=["chat-main"],
        exclude_keywords=("embedding", "tts", "deprecated-legacy"),
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "text-embedding-3"},
            {"id": "chat-main"},
            {"id": "tts-1"},
            {"id": "chat-secondary"},
            {"id": "deprecated-legacy-model"},
        ]
    }

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        models = await provider.list_models()
        assert models == ["chat-main", "chat-secondary"]


# --- Config-driven OpenAI STT model (pattern parity with groq/openrouter profiles) ---

CUSTOM_STT_MODEL = "whisper-large-v3-turbo"


def test_openai_default_stt_model_env_override_and_default():
    # Default must stay "whisper-1" when no override is present
    with patch.dict(os.environ):
        os.environ.pop("OPENAI_DEFAULT_STT_MODEL", None)
        assert Settings().openai_default_stt_model == "whisper-1"

    # Env override configures the model, mirroring GROQ_DEFAULT_STT_MODEL
    with patch.dict(os.environ, {"OPENAI_DEFAULT_STT_MODEL": CUSTOM_STT_MODEL}):
        assert Settings().openai_default_stt_model == CUSTOM_STT_MODEL


def test_openai_engine_profile_honors_configured_default_stt_model():
    original = settings.openai_default_stt_model
    try:
        settings.openai_default_stt_model = CUSTOM_STT_MODEL
        factory = TranscriberFactory()
        engine = factory.get_engine("openai")
        assert engine.default_model == CUSTOM_STT_MODEL
        models = factory.get_models_for_engine("openai")
        assert models[0] == CUSTOM_STT_MODEL
        assert "whisper-1" in models
    finally:
        settings.openai_default_stt_model = original


def test_api_whisper_models_openai_lists_configured_model():
    original = settings.openai_default_stt_model
    try:
        settings.openai_default_stt_model = CUSTOM_STT_MODEL
        with patch("main.transcriber_factory", TranscriberFactory()):
            resp = client.get("/api/whisper/models?engine=openai")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"] == "openai"
        assert data["models"] == [CUSTOM_STT_MODEL, "whisper-1"]
    finally:
        settings.openai_default_stt_model = original


def test_api_whisper_models_openai_default_unchanged_without_override():
    resp = client.get("/api/whisper/models?engine=openai")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "openai"
    assert data["models"] == ["whisper-1"]


def test_openai_compatible_settings_export_includes_default_stt_model():
    cfg = get_masked_settings()
    assert "default_stt_model" in cfg["openai_compatible"]
    assert cfg["openai_compatible"]["default_stt_model"] == settings.openai_default_stt_model


def test_transcription_request_multipart_body_encodes(tmp_path: Path):
    """Regression (httpx>=0.28): the request built for /audio/transcriptions must encode a
    real multipart body. httpx 0.28 raises TypeError when data=list-of-tuples is combined
    with files=, which mocked-post tests never exercise. Build the actual httpx.Request
    from the captured call kwargs and encode it."""
    import httpx

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    transcriber = OpenAICompatibleTranscriber(
        name="openai",
        display_name="OpenAI Cloud Whisper",
        base_url_getter="http://llama-swap:8080/v1",
        api_key_getter="dummy-key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )

    captured: dict = {}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "ok", "language": "en", "duration": 1.0, "segments": []}

    def capture_post(self, url, **kwargs):
        # Encode the real request while the file handle is still open, exactly as a
        # live httpx.Client.post would — this is what raises on httpx 0.28 when
        # data=list-of-tuples is combined with files=.
        req = httpx.Request("POST", url, **kwargs)
        captured["body"] = req.read()
        return mock_response

    with patch("httpx.Client.post", new=capture_post):
        transcriber._call_transcription_api(file_path=audio_file, model_name="whisper-large-v3-turbo")

    body = captured["body"]

    assert b'name="model"' in body
    assert b"whisper-large-v3-turbo" in body
    assert b'name="response_format"' in body
    assert b"verbose_json" in body
    assert body.count(b'name="timestamp_granularities[]"') == 2
    assert b'filename="sample.wav"' in body


def test_language_selector_offers_ukrainian_and_russian():
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'value="auto"' in resp.text
    assert 'value="uk"' in resp.text
    assert 'value="ru"' in resp.text


def test_transcription_request_language_passthrough(tmp_path: Path):
    """Explicit language must be sent in the multipart body; 'auto' must omit it
    so the STT server performs autodetection."""
    import httpx

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    transcriber = OpenAICompatibleTranscriber(
        name="openai",
        display_name="OpenAI Cloud Whisper",
        base_url_getter="http://llama-swap:8080/v1",
        api_key_getter="dummy-key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )

    def capture_body(**call_kwargs) -> bytes:
        captured: dict = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "ok", "language": "uk", "duration": 1.0, "segments": []}

        def capture_post(self, url, **kwargs):
            req = httpx.Request("POST", url, **kwargs)
            captured["body"] = req.read()
            return mock_response

        with patch("httpx.Client.post", new=capture_post):
            transcriber._call_transcription_api(file_path=audio_file, model_name="whisper-large-v3-turbo", **call_kwargs)
        return captured["body"]

    body_uk = capture_body(language="uk")
    assert b'name="language"' in body_uk
    assert b"\r\nuk\r\n" in body_uk

    body_auto = capture_body(language="auto")
    assert b'name="language"' not in body_auto


# ---------------------------------------------------------------------------
# T6 — AH-4 observability (spec §3.4 P4/P7/P8, DEC-8).
# ---------------------------------------------------------------------------

# --- P4: LOG_LEVEL env (default INFO; invalid → INFO) -----------------------

def test_log_level_env_sets_effective_root_level():
    import main

    try:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            main.setup_logging()
            assert logging.getLogger().getEffectiveLevel() == logging.DEBUG

        with patch.dict(os.environ, {"LOG_LEVEL": "NOT_A_LEVEL"}):
            main.setup_logging()
            assert logging.getLogger().getEffectiveLevel() == logging.INFO
    finally:
        os.environ.pop("LOG_LEVEL", None)
        main.setup_logging()
        assert logging.getLogger().getEffectiveLevel() == logging.INFO


# --- P7 (adapter side): on_progress contract --------------------------------

def _make_openai_transcriber() -> OpenAICompatibleTranscriber:
    return OpenAICompatibleTranscriber(
        name="openai",
        display_name="OpenAI Cloud Whisper",
        base_url_getter="http://llama-swap:8080/v1",
        api_key_getter="dummy-key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )


def _ok_response(duration: float) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "text": "ok",
        "language": "en",
        "duration": duration,
        "segments": [],
    }
    return resp


def test_transcribe_on_progress_compressing_phase_precedes_chunks(tmp_path: Path):
    """on_progress is called once with phase='compressing', done=0 BEFORE compression,
    then per chunk with cumulative audio-seconds and phase='chunk'."""
    input_file = tmp_path / "input.mkv"  # non-audio extension forces the compression path
    input_file.write_bytes(b"FAKE-MEDIA")

    transcriber = _make_openai_transcriber()

    def fake_subprocess_run(cmd, **kwargs):
        res = MagicMock()
        if str(cmd[0]).endswith("ffprobe"):
            res.stdout = "90.0"
        else:  # ffmpeg compression: create the (small) output file
            Path(cmd[-1]).write_bytes(b"\x00" * 100)
            res.stdout = ""
        return res

    calls: list[tuple[float, float, str]] = []
    with (
        patch("transcribers.openai_compat.subprocess.run", side_effect=fake_subprocess_run),
        patch("httpx.Client.post", return_value=_ok_response(90.0)),
    ):
        result = transcriber.transcribe(
            input_file,
            on_progress=lambda done, total, phase: calls.append((done, total, phase)),
        )

    assert result.text == "ok"
    assert calls, "on_progress must be called"
    phases = [c[2] for c in calls]
    assert set(phases) <= {"compressing", "chunk"}
    assert phases[0] == "compressing"
    assert calls[0][0] == 0.0
    chunk_calls = [c for c in calls if c[2] == "chunk"]
    assert len(chunk_calls) == 1
    assert chunk_calls[0][0] == pytest.approx(90.0)  # cumulative audio-seconds


def test_transcribe_on_progress_monotonic_cumulative_across_chunks(tmp_path: Path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    chunk1 = tmp_path / "chunk_000.mp3"
    chunk1.write_bytes(b"\x00" * 10)
    chunk2 = tmp_path / "chunk_001.mp3"
    chunk2.write_bytes(b"\x00" * 10)

    transcriber = _make_openai_transcriber()

    def fake_subprocess_run(cmd, **kwargs):
        res = MagicMock()
        if str(cmd[0]).endswith("ffprobe"):
            res.stdout = "60.0"
        else:
            Path(cmd[-1]).write_bytes(b"\x00" * 100)
            res.stdout = ""
        return res

    responses = iter([_ok_response(60.0), _ok_response(60.0)])
    calls: list[tuple[float, float, str]] = []
    with (
        patch("transcribers.openai_compat.MAX_PAYLOAD_BYTES", 50),
        patch("transcribers.openai_compat.subprocess.run", side_effect=fake_subprocess_run),
        patch.object(transcriber, "_split_into_chunks", return_value=[(chunk1, 0.0), (chunk2, 60.0)]),
        patch("httpx.Client.post", side_effect=lambda *a, **k: next(responses)),
    ):
        transcriber.transcribe(
            audio_file,
            on_progress=lambda done, total, phase: calls.append((done, total, phase)),
        )

    chunk_calls = [c for c in calls if c[2] == "chunk"]
    assert len(chunk_calls) == 2
    assert [c[0] for c in chunk_calls] == [60.0, 120.0]  # cumulative, monotonic non-decreasing
    assert all(c[1] == 120.0 for c in chunk_calls)  # total known from chunk plan
    assert all(c[0] <= c[1] for c in chunk_calls)


def test_transcribe_without_on_progress_still_works(tmp_path: Path):
    """on_progress is an optional kwarg — existing call sites unaffected."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
    transcriber = _make_openai_transcriber()
    with patch("httpx.Client.post", return_value=_ok_response(1.0)):
        result = transcriber.transcribe(audio_file)
    assert result.text == "ok"


# --- P8: request timeout from settings (env STT_REQUEST_TIMEOUT, default 600)

def test_stt_request_timeout_setting_default_and_env_override():
    with patch.dict(os.environ):
        os.environ.pop("STT_REQUEST_TIMEOUT", None)
        assert Settings().stt_request_timeout == 600

    with patch.dict(os.environ, {"STT_REQUEST_TIMEOUT": "900"}):
        assert Settings().stt_request_timeout == 900


def test_transcription_request_uses_settings_timeout(tmp_path: Path):
    """The timeout on the REAL request path (httpx client + MockTransport) comes
    from settings — not a constant in source."""
    import httpx

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
    transcriber = _make_openai_transcriber()

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json={"text": "ok", "language": "en", "duration": 1.0, "segments": []})

    real_init = httpx.Client.__init__

    def spy_init(self, *args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    original = settings.stt_request_timeout
    try:
        settings.stt_request_timeout = 424
        with patch.object(httpx.Client, "__init__", spy_init):
            data = transcriber._call_transcription_api(
                file_path=audio_file, model_name="whisper-large-v3-turbo"
            )
        assert data["text"] == "ok"  # the request really went through MockTransport
        assert captured["path"] == "/v1/audio/transcriptions"
        timeout = captured.get("timeout")
        assert timeout is not None, "client must be constructed with an explicit timeout"
        assert timeout.read == 424.0
    finally:
        settings.stt_request_timeout = original



