#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download audio from a JSONL search result file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import anti_scraping


PLATFORM_PREFIX_BY_NAME = {
    "YouTube": "ytsearch",
    "Bilibili": "bilisearch",
    "SoundCloud": "scsearch",
}

MUSIC_BLACKLIST = [
    "mv", "official", "official mv", "official video", "lyrics", "lyric", "翻唱", "cover",
    "歌曲", "歌", "唱歌", "music", "audio", "ost", "live", "remix", "single",
    "ft.", "feat.", "演唱", "演唱会", "音乐会", "舞台", "k歌", "karaoke",
    "现场版", "现场演出", "现场表演", "表演", "热舞", "舞蹈", "原唱", "伴奏",
    "合唱", "专辑", "music video", "musicvideo", "choreography", "performance",
    "singer", "singing", "song", "songs",
]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def entry_blob(entry: dict) -> str:
    parts = [
        entry.get("title") or "",
        entry.get("uploader") or "",
        entry.get("channel") or "",
        entry.get("description") or "",
        entry.get("tags") or "",
    ]
    return normalize_text(" ".join(str(part) for part in parts if part))


def is_music(entry: dict) -> bool:
    blob = entry_blob(entry)
    return any(token in blob for token in MUSIC_BLACKLIST)


def load_entries(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def download_one(entry: dict, out_dir: Path, audio_only: bool = True) -> bool:
    title = entry.get("title") or entry.get("webpage_url") or entry.get("url") or entry.get("id")
    url = entry.get("webpage_url") or entry.get("url") or entry.get("id")
    if not url:
        return False
    platform_name = entry.get("_platform")
    prefix = PLATFORM_PREFIX_BY_NAME.get(platform_name, "ytsearch")
    headers = anti_scraping.get_platform_headers(prefix)

    opts = {
        "format": "bestaudio/best" if audio_only else "best",
        "outtmpl": str(out_dir / "%(title).200s-%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "http_headers": headers,
        "noplaylist": True,
        "retries": 3,
    }
    if audio_only:
        opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "192",
            }
        ]

    try:
        import yt_dlp
    except Exception:
        return False

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
        return True
    except Exception as exc:
        print(f"[skip] {title} -> {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download audio from a search-result JSONL file.")
    parser.add_argument("--input", required=True, help="Path to JSONL search results")
    parser.add_argument("--out-dir", required=True, help="Directory to write downloaded audio")
    parser.add_argument("--limit", type=int, default=None, help="Limit how many entries to download")
    parser.add_argument("--skip-music", action="store_true", default=True, help="Skip obvious music results")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(input_path)
    if args.skip_music:
        entries = [e for e in entries if not is_music(e)]

    downloaded = 0
    for entry in entries:
        if args.limit is not None and downloaded >= args.limit:
            break
        ok = download_one(entry, out_dir, audio_only=True)
        if ok:
            downloaded += 1

    print(f"downloaded={downloaded} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
