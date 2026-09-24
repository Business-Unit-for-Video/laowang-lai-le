"""Validate and normalize a YouTube Netscape cookie export for yt-dlp."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _is_header(line: str) -> bool:
    text = line.strip().lower()
    return text.startswith("# netscape http cookie file") or text.startswith("# http cookie file")


def validate(path: Path) -> tuple[int, int]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("cookie file is missing or empty")
    raw = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    lines = raw.splitlines()
    header_index = next((i for i, line in enumerate(lines) if _is_header(line)), None)
    if header_index is None:
        raise ValueError("cookie file is not a Netscape export")
    lines = lines[header_index:]
    rows = []
    youtube_rows = []
    usable_rows = []
    now = int(time.time())
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        rows.append(line)
        domain = fields[0].replace("#HttpOnly_", "", 1).lower()
        if "youtube.com" not in domain and "youtu.be" not in domain:
            continue
        youtube_rows.append(line)
        try:
            expiry = int(float(fields[4] or "0"))
        except ValueError:
            expiry = -1
        if expiry == 0 or expiry > now:
            usable_rows.append(line)
    if not rows:
        raise ValueError("Netscape export contains no cookie rows")
    if not youtube_rows:
        raise ValueError("Netscape export contains no YouTube cookie rows")
    if not usable_rows:
        raise ValueError("all YouTube cookie rows are expired")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return len(rows), len(usable_rows)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_youtube_cookies.py PATH", file=sys.stderr)
        return 2
    try:
        rows, usable = validate(Path(sys.argv[1]))
    except ValueError as exc:
        print(f"YouTube cookie validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"YouTube cookie validation passed: {rows} rows, {usable} non-expired/session rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
