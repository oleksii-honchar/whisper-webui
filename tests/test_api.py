"""Integration tests for FastAPI endpoints."""

import json
import re
import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Audio Transcriber" in response.text


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ffmpeg"]["available"] is True
    assert len(data["whisper_engines"]) >= 1
    assert len(data["llm_providers"]) >= 1


def test_llm_models_endpoint():
    response = client.get("/api/llm/models?provider=ollama")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "ollama"
    assert "models" in data


def test_api_transcribe_upload():
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "synth.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.5", "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True,
            capture_output=True,
        )

        with open(wav_path, "rb") as f:
            response = client.post(
                "/api/transcribe",
                files={"file": ("synth.wav", f, "audio/wav")},
                data={"whisper_engine": "faster-whisper", "whisper_model": "tiny", "language": "en"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "filename" in data
        assert "text" in data
        assert "srt" in data
        assert "vtt" in data
        assert "segments" in data


# ---------------------------------------------------------------------------
# T7 — AH-5 (spec §3.4 P10) settings round-trip through the generic
# save_config diarization section — AH-18 (DEC-17) re-targeted it to the KEPT
# knobs (engine/api_url/request_timeout/min_speaker_duration); the removed
# local-engine-only knobs no longer exist on Settings.
# ---------------------------------------------------------------------------


def test_settings_roundtrip_flows_diarization_section(monkeypatch, tmp_path: Path):
    import config as config_module

    # Register current singleton values with monkeypatch so save_config's
    # in-memory sync is restored after the test.
    for attr in (
        "diarization_engine",
        "diarization_api_url",
        "diarization_request_timeout",
        "diarization_min_speaker_duration",
    ):
        monkeypatch.setattr(config_module.settings, attr, getattr(config_module.settings, attr))
    monkeypatch.setattr(config_module, "CONFIG_JSON_PATH", tmp_path / "config.json")

    response = client.post(
        "/api/settings",
        json={
            "diarization": {
                "engine": "remote",
                "api_url": "http://sidecar.test:8000",
                "request_timeout": 77,
                "min_speaker_duration": 1.5,
            }
        },
    )
    assert response.status_code == 200

    assert config_module.settings.diarization_engine == "remote"
    assert config_module.settings.diarization_api_url == "http://sidecar.test:8000"
    assert config_module.settings.diarization_request_timeout == 77
    assert config_module.settings.diarization_min_speaker_duration == 1.5

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["diarization"]["engine"] == "remote"
    assert saved["diarization"]["request_timeout"] == 77


def _capture_pipeline_kwargs(monkeypatch) -> tuple[dict, threading.Event]:
    """Replace run_pipeline on the app's job manager with a kwargs recorder."""
    import main as main_module

    captured: dict = {}
    started = threading.Event()

    async def capture_run_pipeline(**kwargs):
        captured.update(kwargs)
        started.set()
        media = kwargs.get("media_path")
        if media and Path(media).exists():
            Path(media).unlink()

    monkeypatch.setattr(main_module.job_manager, "run_pipeline", capture_run_pipeline)
    return captured, started


def test_jobs_endpoint_explicit_cluster_threshold_passes_through(monkeypatch):
    captured, started = _capture_pipeline_kwargs(monkeypatch)

    response = client.post(
        "/api/jobs",
        files={"file": ("t.wav", b"RIFF" + b"\x00" * 32, "audio/wav")},
        data={"ai_action": "raw", "cluster_threshold": "0.3"},
    )
    assert response.status_code == 200
    assert started.wait(timeout=5.0)
    assert captured["cluster_threshold"] == 0.3, "an explicit client value must win"


def test_sync_transcribe_diarization_calls_configured_engine(monkeypatch, tmp_path: Path):
    """AH-18: the sync /transcribe path must call the engine resolved from
    settings — the configured engine's diarize() is the observable contract.
    (Replaces the removed-engine threshold pinning: that setting was deleted
    with the engine, DEC-17; the API param itself stays accepted-and-ignored.)"""
    from transcribers.base import Segment, TranscriptionResult
    from diarization.base import DiarizationResult, SpeakerInterval

    fake_transcriber = MagicMock()
    fake_transcriber.name = "openai"
    fake_transcriber.transcribe.return_value = TranscriptionResult(
        text="hello",
        segments=[Segment(id=0, start=0.0, end=2.0, text="hello")],
        language="en",
        duration=2.0,
    )

    diarizer = MagicMock()
    diarizer.is_available.return_value = True
    diarizer.diarize.return_value = DiarizationResult(
        num_speakers=1,
        intervals=[SpeakerInterval(start=0.0, end=2.0, speaker="spk0")],
    )

    def write_silence_wav(src, dst):
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 32000)  # 2.0 s of audio
        return dst

    audio_processor = MagicMock()
    audio_processor.convert_to_whisper_wav.side_effect = write_silence_wav
    metadata = MagicMock()
    metadata.duration = 2.0
    audio_processor.probe_media.return_value = metadata

    wav_path = tmp_path / "synth.wav"
    write_silence_wav(None, wav_path)

    with (
        patch("main.audio_processor", audio_processor),
        patch("transcribers.transcriber_factory.get_transcriber", return_value=fake_transcriber),
        patch("diarization.diarizer_factory.get_diarizer", return_value=diarizer),
    ):
        with open(wav_path, "rb") as f:
            response = client.post(
                "/api/transcribe",
                files={"file": ("synth.wav", f, "audio/wav")},
                data={"whisper_engine": "openai", "enable_diarization": "true"},
            )

    assert response.status_code == 200
    assert diarizer.diarize.called, (
        "sync /transcribe must call the configured engine's diarize()"
    )
    assert response.json()["num_speakers"] == 1, (
        "the configured engine's result must flow into the response"
    )


# ---------------------------------------------------------------------------
# AH-6/AH-7/AH-8 — Results panel: 50vh cap, pinned header, only the
# viewports scroll.
# AH-6: cap + scroll. AH-7: cap = 50vh ("0.5 of window height").
# AH-8: "only id=\"viewport-transcript\" should be scrollable, but currently
# id=\"results-container\" is scrollabel and header ... is hidden when
# scrolling" → container is a non-scrolling flex column (overflow: hidden),
# #results-header pinned (flex: 0 0 auto), the three viewports own the scroll
# (flex: 1 1 auto; min-height: 0; overflow-y: auto), and the segments list
# drops its nested max-h-[500px] scroller (single scroll owner).
# AH-16 refines the transcript viewport again: controls row pinned (sticky
# top-0, opaque bg), and scroll ownership moves INTO the two content
# containers (#transcript-segments-list / #transcript-plain-text own their
# scrollers; the viewport itself is !overflow-hidden). style.css keeps the
# viewport rules for the other viewports; the transcript overrides via
# Tailwind important modifier.
# Served-asset TestClient pattern (AH-3), contracts refined in place.
# ---------------------------------------------------------------------------


def test_index_page_serves_pinned_header_contract():
    response = client.get("/")
    assert response.status_code == 200
    assert "/static/style.css?v=5.3" in response.text, (
        "index.html must bump the style.css cache-bust version when the scroll-ownership rules ship"
    )
    assert 'id="results-header"' in response.text, (
        "the results header div needs a stable id hook so CSS can pin it"
    )
    segments = re.search(r'<div id="transcript-segments-list"[^>]*>', response.text)
    assert segments, "the transcript segments list div must exist"
    assert "max-h-[500px]" not in segments.group(0), (
        "segments list must not use the legacy fixed-height scroller"
    )
    # AH-16: scroll ownership moved from the viewport to the content containers —
    # the controls row is pinned and the two containers are the only scroll areas.
    assert "overflow-y-auto" in segments.group(0) and "custom-scrollbar" in segments.group(0), (
        "segments list must own its scroller (AH-16: containers are the scroll owners)"
    )
    viewport = re.search(r'<div id="viewport-transcript"[^>]*>', response.text)
    assert viewport and "!overflow-hidden" in viewport.group(0), (
        "the transcript viewport must not scroll as a whole (AH-16)"
    )
    assert re.search(r'class="[^"]*sticky top-0[^"]*bg-slate-900[^"]*"', response.text), (
        "the transcript controls row must be pinned with an opaque background (AH-16)"
    )
    plain = re.search(r'<div id="transcript-plain-text"[^>]*>', response.text)
    assert plain and "overflow-y-auto" in plain.group(0) and "custom-scrollbar" in plain.group(0), (
        "the plain-text container must own its scroller (AH-16)"
    )


def test_served_css_scrolls_only_the_viewports():
    response = client.get("/static/style.css")
    assert response.status_code == 200
    css = response.text

    container = re.search(r"#results-container\s*\{([^}]*)\}", css)
    assert container, "style.css must contain a #results-container rule"
    body = container.group(1)
    assert re.search(r"max-height\s*:\s*50vh", body), (
        "#results-container max-height must stay 50vh (AH-7 contract)"
    )
    assert re.search(r"display\s*:\s*flex", body) and re.search(r"flex-direction\s*:\s*column", body), (
        "#results-container must be a flex column so header and viewport sizes compose"
    )
    assert re.search(r"overflow\s*:\s*hidden", body), (
        "#results-container must clip, not scroll"
    )
    assert not re.search(r"overflow-y\s*:\s*auto", body), (
        "#results-container must NOT scroll itself — only the viewports scroll (AH-8)"
    )

    header = re.search(r"#results-header\s*\{([^}]*)\}", css)
    assert header, "style.css must contain a #results-header rule"
    assert re.search(r"flex\s*:\s*0 0 auto", header.group(1)), (
        "the results header must be pinned (no shrink, no grow)"
    )

    viewports = re.search(
        r"#viewport-transcript[^{,]*(?:,[^{]*)*#viewport-polish[^{,]*(?:,[^{]*)*#viewport-summary[^{]*\{([^}]*)\}",
        css,
    )
    assert viewports, "style.css must size/scroll the three result viewports"
    vbody = viewports.group(1)
    assert re.search(r"flex\s*:\s*1 1 auto", vbody), "viewports must fill the remaining height"
    assert re.search(r"min-height\s*:\s*0", vbody), "flex children need min-height:0 to scroll"
    assert re.search(r"overflow-y\s*:\s*auto", vbody), "the viewports own the scrolling"


# ---------------------------------------------------------------------------
# AH-16c: static asset delivery. The AH-16/AH-16b app.js fixes were invisible
# to the user's browser: the template cache-buster (?v=5.0) was never bumped
# across those changes and /static was served WITHOUT Cache-Control (browsers
# fall back to heuristic freshness and keep running the old app.js).
# Contract: the app.js reference must carry the current cache-bust version, and
# /static responses must force revalidation (Cache-Control: no-cache — the
# existing etag/last-modified still produce cheap 304s).
# style.css keeps ?v=5.3: the asset has not changed since AH-8 bumped it.
# ---------------------------------------------------------------------------


def test_app_js_cache_buster_matches_shipped_asset():
    response = client.get("/")
    assert response.status_code == 200
    assert "/static/app.js?v=5.2" in response.text, (
        "index.html must bump the app.js cache-bust version whenever app.js changes "
        "(AH-16/AH-16b shipped word-renderer fixes under the stale ?v=5.0; "
        "AH-17 bumped to 5.2 for the badge-removal app.js change)"
    )


def test_static_assets_served_with_revalidation():
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-cache", (
        "/static must send Cache-Control: no-cache so browsers revalidate instead of "
        "trusting heuristic freshness (AH-16c: stale cached app.js hid shipped fixes)"
    )
