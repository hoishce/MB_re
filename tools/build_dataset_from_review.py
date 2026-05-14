#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build dataset_build outputs from the latest review-approved clips.

This script does two things:
1. Exports human-approved clean target clips into:
   dataset_build/approved_manual_transcribe/wavs
   dataset_build/approved_manual_transcribe/transcripts
   and writes manifest/train.txt.
2. Copies noisy-but-promising clips into:
   dataset_build/repair_queue/noisy_target/wavs
   and writes a repair manifest for a later repair pass.

The clean clips are transcribed with the repo Whisper CLI and saved as
PCM16/16k/mono WAV + TXT pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "pipeline_temp"
DEFAULT_SOURCE_MANIFEST = PIPELINE_ROOT / "fresh_interview_clean_20260501" / "manifest.jsonl"
DEFAULT_OUT_ROOT = Path(os.environ.get("MOCKINGBIRD_DATASET_BUILD_ROOT", str(REPO_ROOT / "dataset_build")))
DEFAULT_WHISPER_PY = REPO_ROOT / ".venv_advanced" / "Scripts" / "python.exe"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_generated_files(path: Path, patterns: tuple[str, ...]) -> None:
    if not path.exists():
        return
    for pattern in patterns:
        for item in path.glob(pattern):
            if item.is_file() or item.is_symlink():
                item.unlink()


def ensure_pcm16_wav(src: Path, dst: Path) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not dst.exists():
        return False, (proc.stderr or proc.stdout or "")[-1500:]
    return True, ""


def run_whisper_transcribe(
    whisper_python: Path,
    audio_path: Path,
    transcript_path: Path,
    stats_path: Path,
    language: str = "zh",
    beam_size: int = 5,
) -> dict[str, Any]:
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MOCKINGBIRD_PIPELINE_PROFILE"] = "advanced"
    env["PYTHONUTF8"] = "1"

    cmd = [
        str(whisper_python),
        str(REPO_ROOT / "demo" / "whisper.py"),
        "--audio",
        str(audio_path),
        "--output",
        str(transcript_path),
        "--stats-json",
        str(stats_path),
        "--beam-size",
        str(int(beam_size)),
        "--language",
        language,
        "--print-stats",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    stats: dict[str, Any] = {}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = {}
    if proc.returncode not in {0, 2}:
        raise RuntimeError(
            "Whisper transcription failed:\n"
            f"cmd={' '.join(cmd)}\n"
            f"return_code={proc.returncode}\n"
            f"stdout={proc.stdout[-1500:]}\n"
            f"stderr={proc.stderr[-1500:]}"
        )
    return {
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stats": stats,
    }


def transcript_is_valid(text: str, no_speech_prob: float | None = None) -> tuple[bool, str]:
    text = (text or "").strip()
    if len(text) < 2:
        return False, "text_too_short"
    bad_phrases = [
        "谢谢观看",
        "字幕",
        "由amara",
        "欢迎订阅",
        "请不吝点赞",
        "优优独播剧场",
        "exclusive",
    ]
    normalized = text.replace(" ", "").lower()
    for phrase in bad_phrases:
        if phrase.replace(" ", "").lower() in normalized:
            return False, "asr_hallucination"
    if no_speech_prob is not None and no_speech_prob > 0.60:
        return False, "no_speech_prob_high"
    return True, "ok"


def get_source_path(record: dict[str, Any]) -> Path:
    for key in ("current_path", "path_for_verify", "path_for_vad", "path", "source_path"):
        value = record.get(key)
        if value:
            return Path(str(value))
    raise FileNotFoundError("No usable source path in record")


def export_clean_manual_transcribe(
    records: list[dict[str, Any]],
    out_root: Path,
    whisper_python: Path,
) -> list[dict[str, Any]]:
    clean_root = out_root / "approved_manual_transcribe"
    wav_dir = clean_root / "wavs"
    txt_dir = clean_root / "transcripts"
    manifest_path = clean_root / "manifest.jsonl"
    train_txt_path = clean_root / "train.txt"
    for folder in (wav_dir, txt_dir):
        folder.mkdir(parents=True, exist_ok=True)
        clear_generated_files(folder, ("*.wav", "*.txt"))
    for path in (manifest_path, train_txt_path):
        if path.exists():
            path.unlink()

    scratch_root = out_root / "_scratch" / "manual_transcribe"
    scratch_root.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    train_lines: list[str] = []

    for idx, record in enumerate(records, 1):
        src = get_source_path(record).resolve()
        sample_id = f"{idx:06d}"
        out_wav = wav_dir / f"{sample_id}.wav"
        out_txt = txt_dir / f"{sample_id}.txt"
        scratch_txt = scratch_root / f"{sample_id}.txt"
        scratch_stats = scratch_root / f"{sample_id}.stats.json"

        ok, err = ensure_pcm16_wav(src, out_wav)
        if not ok:
            raise RuntimeError(f"PCM16 export failed for {src}: {err}")

        whisper_info = run_whisper_transcribe(
            whisper_python=whisper_python,
            audio_path=out_wav,
            transcript_path=scratch_txt,
            stats_path=scratch_stats,
            language="zh",
            beam_size=5,
        )
        stats = whisper_info.get("stats", {})
        text = ""
        if scratch_txt.exists():
            text = scratch_txt.read_text(encoding="utf-8").strip()

        valid, reason = transcript_is_valid(text, stats.get("no_speech_prob"))
        if not valid:
            # Keep the artifact, but surface the issue in the manifest.
            reason = f"whisper_{reason}"

        out_txt.write_text(text, encoding="utf-8")
        train_lines.append(f"{sample_id}.wav|{text}")

        exported.append(
            {
                "sample_id": sample_id,
                "segment_id": record.get("segment_id"),
                "source_path": str(src.resolve().as_posix()),
                "wav_path": str(out_wav.resolve().as_posix()),
                "txt_path": str(out_txt.resolve().as_posix()),
                "bucket": "approved_manual_transcribe",
                "human_label": "clean_target",
                "human_confirmed": True,
                "needs_transcript": True,
                "manual_override": True,
                "override_reason": "human_confirmed_clean_target",
                "whisper_text": text,
                "whisper_valid": bool(valid),
                "whisper_reason": reason,
                "whisper_no_speech_prob": stats.get("no_speech_prob"),
                "whisper_avg_logprob": stats.get("avg_logprob"),
                "whisper_language": stats.get("language"),
                "whisper_confidence": stats.get("quality", {}).get("confidence"),
            }
        )

    write_jsonl(exported, manifest_path)
    train_txt_path.write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    return exported


def export_repair_queue(records: list[dict[str, Any]], out_root: Path) -> list[dict[str, Any]]:
    repair_root = out_root / "repair_queue" / "noisy_target"
    wav_dir = repair_root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_files(wav_dir, ("*.wav",))
    manifest_path = repair_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    exported: list[dict[str, Any]] = []
    for idx, record in enumerate(records, 1):
        src = get_source_path(record).resolve()
        sample_id = f"{idx:06d}"
        out_wav = wav_dir / f"{sample_id}.wav"
        ok, err = ensure_pcm16_wav(src, out_wav)
        if not ok:
            exported.append(
                {
                    "sample_id": sample_id,
                    "segment_id": record.get("segment_id"),
                    "source_path": str(src.resolve().as_posix()),
                    "wav_path": str(out_wav.resolve().as_posix()),
                    "bucket": "repair_queue",
                    "repair_status": "copy_failed",
                    "repair_error": err,
                    "final_reject_reason": ";".join(record.get("final_reasons") or []),
                    "music_score": record.get("music_score"),
                    "speaker_similarity": record.get("speaker_similarity"),
                }
            )
            continue
        exported.append(
            {
                "sample_id": sample_id,
                "segment_id": record.get("segment_id"),
                "source_path": str(src.resolve().as_posix()),
                "wav_path": str(out_wav.resolve().as_posix()),
                "bucket": "repair_queue",
                "human_label": "noisy_target",
                "human_confirmed": True,
                "repair_status": "queued",
                "final_reject_reason": ";".join(record.get("final_reasons") or []),
                "music_score": record.get("music_score"),
                "speaker_similarity": record.get("speaker_similarity"),
            }
        )

    write_jsonl(exported, manifest_path)
    return exported


def load_latest_review_manifest(manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_jsonl(manifest_path)
    manual = [r for r in rows if r.get("final_bucket") == "manual_transcribe"]
    noisy = [
        r
        for r in rows
        if r.get("final_bucket") == "reject"
        and any(reason == "music_score_high" for reason in (r.get("final_reasons") or []))
    ]
    return manual, noisy


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset_build from the latest review outputs.")
    parser.add_argument("--source-manifest", type=str, default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--whisper-python", type=str, default=str(DEFAULT_WHISPER_PY))
    parser.add_argument("--run-repair-pipeline", action="store_true", help="After queueing noisy clips, run the repair pipeline on them.")
    parser.add_argument("--speaker-ref", type=str, default=str(REPO_ROOT / "references"))
    args = parser.parse_args()

    source_manifest = Path(args.source_manifest)
    out_root = Path(args.out_root)
    whisper_python = Path(args.whisper_python)
    speaker_ref = Path(args.speaker_ref)
    out_root.mkdir(parents=True, exist_ok=True)

    manual_records, noisy_records = load_latest_review_manifest(source_manifest)
    if not manual_records and not noisy_records:
        print(f"No review records found in {source_manifest}")
        return 1

    print(f"manual_transcribe={len(manual_records)} noisy_target={len(noisy_records)}")
    clean_export = export_clean_manual_transcribe(manual_records, out_root, whisper_python)
    repair_export = export_repair_queue(noisy_records, out_root)

    if args.run_repair_pipeline and noisy_records:
        repair_input = out_root / "repair_queue" / "noisy_target" / "wavs"
        repair_out = out_root / "repair_output"
        env = os.environ.copy()
        env["MOCKINGBIRD_PIPELINE_PROFILE"] = "advanced"
        env["PYTHONUTF8"] = "1"
        cmd = [
            str(whisper_python),
            str(REPO_ROOT / "demo" / "cleaning_pipeline_v2.py"),
            "--input-dir",
            str(repair_input),
            "--out-root",
            str(repair_out),
            "--speaker-ref",
            str(speaker_ref),
            "--enable-demucs",
            "--enable-pyannote",
            "--enable-voiceprint",
            "--enable-whisper",
            "--demucs-batch-size",
            "1",
            "--pyannote-batch-size",
            "1",
            "--voiceprint-batch-size",
            "1",
            "--whisper-batch-size",
            "1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"repair pipeline failed with return code {proc.returncode}")
        print(proc.stdout)
        print(f"repair pipeline complete -> {repair_out}")

    print(f"approved_manual_transcribe exported: {len(clean_export)}")
    print(f"repair_queue queued: {len(repair_export)}")
    print(f"dataset root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
