from pathlib import Path

def test_laowang_corpus_and_state_are_present():
    assert Path("scripts/transcribe_youtube_channel.py").is_file()
    assert Path("scripts/transcription_integrity.py").is_file()
    assert Path("state_youtube/老王来了/progress.json").is_file()
    assert Path("youtube_channels/老王来了/_manifest.json").is_file()
