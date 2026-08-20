import importlib.util
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

fake_faster_whisper = types.ModuleType("faster_whisper")
fake_faster_whisper.WhisperModel = object
sys.modules.setdefault("faster_whisper", fake_faster_whisper)

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from transcription_integrity import atomic_write_text_pair


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Segment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class FakeModel:
    def __init__(self):
        self.calls = 0

    def transcribe(self, _path, **_kwargs):
        self.calls += 1
        info = types.SimpleNamespace(language="zh", language_probability=1)
        return iter([Segment(0, 1799 if self.calls < 3 else 1045, f"part {self.calls}")]), info


def fake_ffmpeg_run(cmd, **_kwargs):
    if cmd[0] == "ffmpeg":
        Path(cmd[-1]).write_bytes(b"RIFF-test-wave")
    return types.SimpleNamespace(stdout="")


class LegacyIntegrityTests(unittest.TestCase):
    def test_all_legacy_entrypoints_default_to_one_whole_call(self):
        for name in ("transcribe_bili", "transcribe_bili_collection", "transcribe_youtube_channel"):
            module = load_script(name)
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                model = FakeModel()
                result = module.transcribe_audio(model, Path("audio.mp3"), 2 * 3600 + 1)
                self.assertEqual(model.calls, 1)
                self.assertEqual(result["segments"], 1)

    def test_rejects_partial_result_before_writing(self):
        module = load_script("transcribe_bili_collection")
        result = {"segments": 1, "plain_text": "partial", "audio_duration": 4646, "transcript_end": 2401}
        with self.assertRaisesRegex(RuntimeError, "transcription is incomplete"):
            module.validate_transcription(result)

    def test_ffprobe_duration_and_truncated_download(self):
        module = load_script("transcribe_bili")
        fake_run = lambda *_args, **_kwargs: types.SimpleNamespace(stdout="3600.5")
        self.assertAlmostEqual(module.probe_audio_duration(Path("x.mp3"), fake_run), 3600.5)
        with self.assertRaisesRegex(RuntimeError, "downloaded audio is incomplete"):
            module.validate_download_duration({"duration": 3600}, 1000)

    def test_rejects_non_finite_duration_and_segment_timestamp(self):
        module = load_script("transcribe_bili_collection")
        for value in ("nan", "inf", "-inf"):
            with self.subTest(duration=value), self.assertRaisesRegex(RuntimeError, "invalid audio duration"):
                module.probe_audio_duration(
                    Path("x.mp3"), lambda *_args, **_kwargs: types.SimpleNamespace(stdout=value)
                )

        class BadModel:
            def transcribe(self, _path, **_kwargs):
                info = types.SimpleNamespace(language="zh", language_probability=1)
                return iter([Segment(0, math.inf, "bad")]), info

        with self.assertRaisesRegex(RuntimeError, "non-finite segment timestamp"):
            module.transcribe_audio(BadModel(), Path("audio.mp3"), 100)

    def test_chunk_fallback_is_explicit_opt_in(self):
        module = load_script("transcribe_youtube_channel")
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "nested" / "tmp"
            with patch.object(module, "TRANSCRIBE_CHUNKED", True), patch.object(module, "TMP_DIR", missing), patch.object(module, "run"):
                with self.assertRaisesRegex(RuntimeError, "did not create a usable audio chunk"):
                    module.transcribe_audio(FakeModel(), Path("audio.mp3"), 60)
            self.assertTrue(missing.is_dir())

    def test_rejects_timestamp_outside_final_chunk(self):
        module = load_script("transcribe_youtube_channel")

        class OverflowModel:
            def transcribe(self, _path, **_kwargs):
                info = types.SimpleNamespace(language="zh", language_probability=1)
                return iter([Segment(0, 1000, "text")]), info

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(module, "TMP_DIR", Path(temp_dir)):
                with patch.object(module, "run", side_effect=fake_ffmpeg_run):
                    with self.assertRaisesRegex(RuntimeError, "outside the audio"):
                        module.transcribe_audio(OverflowModel(), Path("audio.mp3"), 100)

    def test_two_hour_timestamp_does_not_wrap(self):
        module = load_script("transcribe_youtube_channel")
        self.assertEqual(module.seconds_to_mmss_mmm(2 * 3600 + 5.125), "120:05.125")
        bili = load_script("transcribe_bili")
        self.assertEqual(bili.seconds_to_hms(2 * 3600 + 5), "02:00:05")

    def test_malformed_source_duration_is_ignored(self):
        module = load_script("transcribe_bili")
        for value in ("unknown", {}, math.nan, math.inf, -1):
            with self.subTest(duration=value):
                module.validate_download_duration({"duration": value}, 3600)

    def test_collection_queue_preserves_source_duration(self):
        module = load_script("transcribe_bili_collection")
        queue = module.build_queue_from_entries([{
            "id": "BV123", "url": "BV123", "title": "title", "duration": "7200.5",
        }])
        self.assertEqual(queue[0]["duration"], "7200.5")

    def test_force_retranscribe_ignores_existing_outputs(self):
        module = load_script("transcribe_youtube_channel")
        item = {"id": "video-1", "title": "title"}
        with patch.object(module, "is_item_completed", return_value=True):
            with patch.object(module, "FORCE_RETRANSCRIBE", False):
                self.assertIsNone(module.find_next_item([item], set(), set()))
            with patch.object(module, "FORCE_RETRANSCRIBE", True):
                self.assertEqual(module.find_next_item([item], set(), set()), item)

    def test_pair_write_does_not_touch_old_files_when_staging_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "ts" / "item.txt"
            second = Path(temp_dir) / "plain" / "item.txt"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("old ts", encoding="utf-8")
            second.write_text("old plain", encoding="utf-8")
            real_named_temp = tempfile.NamedTemporaryFile
            calls = 0

            def fail_second_stage(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("disk full")
                return real_named_temp(*args, **kwargs)

            with patch("transcription_integrity.tempfile.NamedTemporaryFile", side_effect=fail_second_stage):
                with self.assertRaisesRegex(OSError, "disk full"):
                    atomic_write_text_pair(first, "new ts", second, "new plain")
            self.assertEqual(first.read_text(encoding="utf-8"), "old ts")
            self.assertEqual(second.read_text(encoding="utf-8"), "old plain")


if __name__ == "__main__":
    unittest.main()
