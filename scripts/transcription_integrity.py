"""Shared long-audio transcription integrity helpers for legacy workflows."""

import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict


def probe_audio_duration(audio_path: Path, run: Callable) -> float:
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ], capture=True)
    try:
        duration = float(result.stdout.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        stdout = getattr(result, "stdout", None)
        raise RuntimeError(f"unable to read audio duration from ffprobe: {stdout!r}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"invalid audio duration: {duration}")
    return duration


def validate_download_duration(item: Dict, audio_duration: float):
    """Reject a clearly truncated download when yt-dlp supplied source duration."""
    expected = item.get("duration")
    if not expected:
        return
    try:
        expected = float(expected)
    except (TypeError, ValueError):
        return
    if not math.isfinite(expected) or expected <= 0:
        return
    allowed_shortfall = max(10.0, expected * 0.02)
    if audio_duration < expected - allowed_shortfall:
        raise RuntimeError(
            "downloaded audio is incomplete: "
            f"audio={audio_duration:.3f}s, source={expected:.3f}s, "
            f"shortfall={expected - audio_duration:.3f}s"
        )


def transcribe_audio_chunked(
    model,
    audio_path: Path,
    audio_duration: float,
    tmp_dir: Path,
    run: Callable,
    transcribe_kwargs: Dict,
    timestamp_formatter: Callable[[float], str],
    chunk_seconds: int = 1800,
) -> Dict:
    """Transcribe every chunk and convert chunk-relative timestamps to global time."""
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise RuntimeError(f"invalid audio duration: {audio_duration}")

    ts_lines = []
    plain_lines = []
    kept_segments = 0
    transcript_end = 0.0
    detected_info = None
    chunk_seconds = max(60, int(chunk_seconds))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="transcribe_chunks_", dir=tmp_dir) as chunk_dir:
        chunk_start = 0.0
        chunk_index = 0
        while chunk_start < audio_duration:
            chunk_duration = min(float(chunk_seconds), audio_duration - chunk_start)
            chunk_path = Path(chunk_dir) / f"chunk_{chunk_index:04d}.wav"
            print(
                f"[info] transcribing chunk {chunk_index + 1}: "
                f"{chunk_start:.3f}s..{chunk_start + chunk_duration:.3f}s",
                flush=True,
            )
            run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{chunk_start:.3f}", "-t", f"{chunk_duration:.3f}",
                "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(chunk_path),
            ])
            if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg did not create a usable audio chunk: {chunk_path}")

            segments, current_info = model.transcribe(str(chunk_path), **transcribe_kwargs)
            if detected_info is None:
                detected_info = current_info
            try:
                for segment in segments:
                    text = (segment.text or "").strip()
                    if not text:
                        continue
                    segment_start = float(segment.start)
                    segment_end = float(segment.end)
                    if not math.isfinite(segment_start) or not math.isfinite(segment_end):
                        raise RuntimeError("transcription produced a non-finite segment timestamp")
                    if segment_end < segment_start:
                        raise RuntimeError("transcription produced a segment ending before it starts")
                    if segment_start > chunk_duration + 5.0 or segment_end > chunk_duration + 5.0:
                        raise RuntimeError("transcription produced a timestamp outside its audio chunk")
                    start = min(audio_duration, chunk_start + max(0.0, segment_start))
                    end = min(audio_duration, chunk_start + max(0.0, segment_end))
                    kept_segments += 1
                    transcript_end = max(transcript_end, end)
                    ts_lines.append(
                        f"[{timestamp_formatter(start)} --> {timestamp_formatter(end)}] {text}"
                    )
                    plain_lines.append(text)
            finally:
                chunk_path.unlink(missing_ok=True)
            chunk_start += chunk_duration
            chunk_index += 1

    return {
        "language": getattr(detected_info, "language", ""),
        "language_probability": getattr(detected_info, "language_probability", ""),
        "segments": kept_segments,
        "audio_duration": audio_duration,
        "transcript_end": transcript_end,
        "coverage_ratio": transcript_end / audio_duration,
        "timestamp_text": "\n".join(ts_lines).strip(),
        "plain_text": "\n".join(plain_lines).strip(),
    }


def transcribe_audio_whole(model, audio_path: Path, audio_duration: float,
                           transcribe_kwargs: Dict,
                           timestamp_formatter: Callable[[float], str]) -> Dict:
    """Transcribe the complete audio in one call and retain its global timestamps."""
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise RuntimeError(f"invalid audio duration: {audio_duration}")

    segments, detected_info = model.transcribe(str(audio_path), **transcribe_kwargs)
    ts_lines = []
    plain_lines = []
    kept_segments = 0
    transcript_end = 0.0
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        start = float(segment.start)
        end = float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise RuntimeError("transcription produced a non-finite segment timestamp")
        if end < start:
            raise RuntimeError("transcription produced a segment ending before it starts")
        if start > audio_duration + 5.0 or end > audio_duration + 5.0:
            raise RuntimeError("transcription produced a timestamp outside the audio")
        start = min(audio_duration, max(0.0, start))
        end = min(audio_duration, max(0.0, end))
        kept_segments += 1
        transcript_end = max(transcript_end, end)
        ts_lines.append(f"[{timestamp_formatter(start)} --> {timestamp_formatter(end)}] {text}")
        plain_lines.append(text)

    return {
        "language": getattr(detected_info, "language", ""),
        "language_probability": getattr(detected_info, "language_probability", ""),
        "segments": kept_segments,
        "audio_duration": audio_duration,
        "transcript_end": transcript_end,
        "coverage_ratio": transcript_end / audio_duration,
        "timestamp_text": "\n".join(ts_lines).strip(),
        "plain_text": "\n".join(plain_lines).strip(),
    }


def validate_transcription(
    result: Dict,
    max_trailing_gap_seconds: float = 120.0,
    max_trailing_gap_ratio: float = 0.10,
):
    audio_duration = float(result.get("audio_duration") or 0)
    transcript_end = float(result.get("transcript_end") or 0)
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise RuntimeError("transcription has no valid audio duration")
    if not math.isfinite(transcript_end) or transcript_end < 0:
        raise RuntimeError("transcription has an invalid end timestamp")
    if not result.get("segments") or not result.get("plain_text"):
        raise RuntimeError("transcription produced no text")
    trailing_gap = max(0.0, audio_duration - transcript_end)
    allowed_gap = max(float(max_trailing_gap_seconds), audio_duration * float(max_trailing_gap_ratio))
    if trailing_gap > allowed_gap:
        raise RuntimeError(
            "transcription is incomplete: "
            f"audio={audio_duration:.3f}s, transcript_end={transcript_end:.3f}s, "
            f"trailing_gap={trailing_gap:.3f}s, allowed={allowed_gap:.3f}s"
        )
    print(
        f"[ok] transcription coverage: {transcript_end:.3f}/{audio_duration:.3f}s "
        f"({transcript_end / audio_duration:.2%}), trailing gap {trailing_gap:.3f}s",
        flush=True,
    )


def atomic_write_text_pair(first_path: Path, first_content: str, second_path: Path, second_content: str):
    """Stage both outputs before replacing either, and restore old files on replace failure."""
    paths_and_content = ((first_path, first_content), (second_path, second_content))
    staged = []
    backups = []
    replaced = []
    try:
        for path, content in paths_and_content:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append(Path(handle.name))

        for path, _content in paths_and_content:
            if path.exists():
                with tempfile.NamedTemporaryFile(
                    prefix=f".{path.name}.", suffix=".bak", dir=path.parent, delete=False,
                ) as handle:
                    backup = Path(handle.name)
                shutil.copy2(path, backup)
                backups.append(backup)
            else:
                backups.append(None)

        for index, (path, _content) in enumerate(paths_and_content):
            staged[index].replace(path)
            replaced.append(index)
    except Exception:
        for index in reversed(replaced):
            path = paths_and_content[index][0]
            backup = backups[index]
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                backup.replace(path)
                backups[index] = None
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
        for path in backups:
            if path is not None:
                path.unlink(missing_ok=True)
