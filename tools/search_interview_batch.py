#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch search interview / variety candidates with music filtering."""

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


PLATFORMS = {
    "ytsearch": "YouTube",
    "bilisearch": "Bilibili",
    "scsearch": "SoundCloud",
}

INTERVIEW_HINTS = [
    "访谈", "采访", "专访", "对谈", "对话", "脱口秀", "圆桌", "聊天", "嘉宾", "人物", "故事",
]

INTERVIEW_SHOW_SEEDS = [
    "鲁豫有约", "圆桌派", "十三邀", "开讲啦", "面对面", "金星秀", "今晚80后脱口秀", "晓松奇谈", "天天向上",
]

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


def build_queries(seed: str) -> list[str]:
    base = (seed or "").strip()
    queries = [base] if base else []
    for hint in INTERVIEW_HINTS:
        queries.append(f"{base} {hint}".strip())
    for show in INTERVIEW_SHOW_SEEDS:
        queries.append(f"{base} {show}".strip())
    queries.extend([
        f"{base} 访谈节目".strip(),
        f"{base} 综艺访谈".strip(),
        f"{base} 人物访谈".strip(),
        f"{base} 嘉宾对谈".strip(),
    ])
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        query = query.strip()
        if query and query not in seen:
            seen.add(query)
            deduped.append(query)
    return deduped


def dedupe(entries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for entry in entries:
        key = str(entry.get("webpage_url") or entry.get("url") or entry.get("id") or entry.get("title") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def search_batch(platform_keys: list[str], seeds: list[str], per_query: int, max_results: int) -> list[dict]:
    collected: list[dict] = []
    for platform_key in platform_keys:
        platform_name = PLATFORMS[platform_key]
        headers = anti_scraping.get_platform_headers(platform_key)
        for seed in seeds:
            for query in build_queries(seed):
                try:
                    batch = anti_scraping.yt_dlp_search_for_platform(platform_key, per_query, query, http_headers=headers)
                except Exception:
                    continue
                if not batch:
                    continue
                batch = [e for e in batch if not is_music(e)]
                for item in batch:
                    item["_platform"] = platform_name
                    item["_query"] = query
                collected.extend(batch)
                collected = dedupe(collected)
                if len(collected) >= max_results:
                    return collected[:max_results]
    collected.sort(key=lambda e: (len(entry_blob(e)), e.get("title") or ""))
    return collected[:max_results]


def main() -> int:
    parser = argparse.ArgumentParser(description="Search interview / variety candidates.")
    parser.add_argument("--out", default=str(REPO_ROOT / "pipeline_temp" / "interview_candidates.jsonl"))
    parser.add_argument("--per-query", type=int, default=5)
    parser.add_argument("--max-results", type=int, default=60)
    parser.add_argument("--platform", action="append", choices=sorted(PLATFORMS.keys()), help="Limit platforms")
    parser.add_argument("--seed", action="append", help="Add a custom seed term")
    args = parser.parse_args()

    platform_keys = args.platform or list(PLATFORMS.keys())
    seeds = args.seed or [
        "访谈", "采访", "脱口秀", "对谈", "圆桌派", "鲁豫有约", "十三邀", "开讲啦", "面对面", "金星秀",
    ]
    results = search_batch(platform_keys, seeds, args.per_query, args.max_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in results) + ("\n" if results else ""),
        encoding="utf-8",
    )

    print(f"saved={len(results)} -> {out_path}")
    for idx, item in enumerate(results[:20], 1):
        print(f"{idx:02d}. [{item.get('_platform')}] {item.get('title')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
