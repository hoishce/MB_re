from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_BUILD_ROOT = Path(os.environ.get("MOCKINGBIRD_DATASET_BUILD_ROOT", str(REPO_ROOT / "dataset_build")))
ADVANCED_PYTHON = Path(os.environ.get("MOCKINGBIRD_ADVANCED_PYTHON", str(REPO_ROOT / ".venv_advanced" / "Scripts" / "python.exe")))


def load_turns(diarized_json: Path) -> list[dict]:
    payload = json.loads(diarized_json.read_text(encoding="utf-8"))
    turns = payload.get("turns") or []
    out: list[dict] = []
    for turn in turns:
        try:
            out.append(
                {
                    "start": float(turn.get("start", 0.0)),
                    "end": float(turn.get("end", 0.0)),
                    "speaker": str(turn.get("speaker", "")),
                }
            )
        except Exception:
            continue
    return sorted(out, key=lambda x: (x["start"], x["end"]))


def merge_turns(turns: Iterable[dict], pad_sec: float, duration: float) -> tuple[float, float] | None:
    selected = list(turns)
    if not selected:
        return None
    start = max(0.0, float(selected[0]["start"]) - pad_sec)
    end = min(duration, float(selected[-1]["end"]) + pad_sec)
    if end <= start:
        return None
    return start, end


def crop_wav(src: Path, start_sec: float, end_sec: float, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(src), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    start = max(0, int(round(start_sec * sr)))
    end = min(len(data), int(round(end_sec * sr)))
    if end <= start:
        return False
    sf.write(str(dst), data[start:end], sr, subtype="PCM_16")
    return dst.exists()


def run_whisper(audio_path: Path, output_txt: Path, stats_json: Path, whisper_py: Path, whisper_env: Path) -> dict:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MOCKINGBIRD_PIPELINE_PROFILE"] = "advanced"
    cmd = [
        str(whisper_env),
        str(whisper_py),
        "--audio",
        str(audio_path),
        "--output",
        str(output_txt),
        "--stats-json",
        str(stats_json),
        "--beam-size",
        "5",
        "--language",
        "zh",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    result = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if stats_json.exists():
        try:
            result["stats"] = json.loads(stats_json.read_text(encoding="utf-8"))
        except Exception:
            result["stats"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop target-speaker intervals from review audio and transcribe them.")
    parser.add_argument("--approved-source", default=str(DATASET_BUILD_ROOT / "approved_manual_transcribe" / "wavs" / "000002.wav"))
    parser.add_argument("--repairs-source-dir", default=str(DATASET_BUILD_ROOT / "repair_queue" / "noisy_target" / "wavs"))
    parser.add_argument("--diarized-root", default=str(DATASET_BUILD_ROOT / "crop_run_20260502" / "diarized"))
    parser.add_argument("--source-manifest", default=str(DATASET_BUILD_ROOT / "crop_run_20260502" / "manifest.jsonl"))
    parser.add_argument("--output-root", default=str(DATASET_BUILD_ROOT / "crop_stage_20260502"))
    parser.add_argument("--whisper-py", default=str(REPO_ROOT / "demo" / "whisper.py"))
    parser.add_argument("--whisper-env", default=str(ADVANCED_PYTHON))
    parser.add_argument("--pad-sec", type=float, default=0.2)
    parser.add_argument("--midpoint-threshold", type=float, default=4.5)
    args = parser.parse_args()

    approved_source = Path(args.approved_source)
    repairs_source_dir = Path(args.repairs_source_dir)
    diarized_root = Path(args.diarized_root)
    source_manifest = Path(args.source_manifest)
    output_root = Path(args.output_root)
    whisper_py = Path(args.whisper_py)
    whisper_env = Path(args.whisper_env)

    out_manifest = []

    jobs: list[tuple[str, Path, Path, float | None, Path | None]] = []
    if approved_source.exists():
        jobs.append(
            (
                "approved_manual_transcribe",
                approved_source,
                output_root / "approved_manual_transcribe" / "wavs" / approved_source.name,
                args.midpoint_threshold,
                None,
            )
        )

    repair_manifest_rows = []
    if source_manifest.exists():
        try:
            repair_manifest_rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            repair_manifest_rows = []
    repair_rows = [row for row in repair_manifest_rows if str(row.get("source_path", "")).lower().endswith(".wav") and "repair_" in str(row.get("source_path", "")).lower()]
    repair_rows = [row for row in repair_rows if str(row.get("source_path", "")).split("\\")[-1].lower().startswith("repair_")]
    repair_rows.sort(key=lambda row: str(row.get("source_path", "")))
    repair_diarized: list[Path] = []
    for row in repair_rows:
        source_name = Path(str(row.get("source_path", ""))).stem
        candidates = sorted(diarized_root.glob(f"{source_name}*.json"))
        if candidates:
            repair_diarized.append(candidates[0])

    repair_sources = sorted(repairs_source_dir.glob("*.wav"))
    for idx, src in enumerate(repair_sources):
        mapped = repair_diarized[idx] if idx < len(repair_diarized) else None
        jobs.append(
            (
                "repair_queue/noisy_target",
                src,
                output_root / "repair_queue" / "noisy_target" / "wavs" / src.name,
                None,
                mapped,
            )
        )

    for category, src, dst, midpoint_threshold, diarized_override in jobs:
        stem = src.stem
        diar_json_candidates = [diarized_override] if diarized_override else list(diarized_root.glob(f"{stem}*.json"))
        diar_json_candidates = [p for p in diar_json_candidates if p and p.exists()]
        if not diar_json_candidates:
            out_manifest.append({
                "source": str(src),
                "category": category,
                "status": "missing_diarization",
            })
            continue
        diar_json = sorted(diar_json_candidates)[0]
        turns = load_turns(diar_json)
        info = sf.info(str(src))
        duration = float(info.frames) / float(info.samplerate or 1)

        if midpoint_threshold is not None:
            late_turns = [t for t in turns if float(t["start"]) >= midpoint_threshold]
            selected_turns = late_turns or turns
        else:
            selected_turns = turns

        crop = merge_turns(selected_turns, args.pad_sec, duration)
        if crop is None:
            out_manifest.append({
                "source": str(src),
                "category": category,
                "status": "no_crop_interval",
            })
            continue

        start_sec, end_sec = crop
        crop_dir = dst.parent
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_ok = crop_wav(src, start_sec, end_sec, dst)
        if not crop_ok:
            out_manifest.append({
                "source": str(src),
                "category": category,
                "status": "crop_failed",
                "start_sec": start_sec,
                "end_sec": end_sec,
            })
            continue

        txt_path = dst.with_suffix(".txt").with_name(dst.stem + ".txt")
        stats_path = dst.with_suffix(".json").with_name(dst.stem + ".stats.json")
        whisper_result = run_whisper(dst, txt_path, stats_path, whisper_py, whisper_env)
        stats = whisper_result.get("stats") or {}
        text = str(stats.get("text", "") or "").strip()
        out_manifest.append(
            {
                "source": str(src),
                "category": category,
                "status": "cropped_and_transcribed",
                "diarized_json": str(diar_json),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "output_wav": str(dst),
                "output_txt": str(txt_path),
                "text": text,
                "audio_duration_sec": stats.get("audio_duration_sec"),
                "transcript_duration_sec": stats.get("transcript_duration_sec"),
                "segment_count": stats.get("segment_count"),
                "avg_logprob": stats.get("avg_logprob"),
                "no_speech_prob": stats.get("no_speech_prob"),
                "quality": stats.get("quality"),
                "whisper_returncode": whisper_result.get("returncode"),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in out_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "cropped": sum(1 for row in out_manifest if row.get("status") == "cropped_and_transcribed"),
        "other": len(out_manifest) - sum(1 for row in out_manifest if row.get("status") == "cropped_and_transcribed"),
        "manifest": str(manifest_path),
        "output_root": str(output_root),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
