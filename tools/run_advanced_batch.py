#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the advanced MockingBird batch in chunked mode and merge results."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ADV_PYTHON = Path(
    os.environ.get("MOCKINGBIRD_ADVANCED_PYTHON", str(REPO_ROOT / ".venv_advanced" / "Scripts" / "python.exe"))
)
PIPELINE_PY = REPO_ROOT / "demo" / "cleaning_pipeline_v2.py"
CHUNK_DIR = REPO_ROOT / "pipeline_temp" / "batch_advanced_subchunks"
RUN_ROOT = REPO_ROOT / "pipeline_temp" / "batch_advanced_subruns"
MERGED_ROOT = REPO_ROOT / "pipeline_temp" / "batch_advanced_final_merged"
LOG_PATH = RUN_ROOT / "run.log"


def log(message: str) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")
    print(message, flush=True)


def run_chunk(chunk_path: Path, out_root: Path) -> int:
    env = os.environ.copy()
    env["MOCKINGBIRD_PIPELINE_PROFILE"] = "advanced"
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.pop("PYTHONPATH", None)
    cmd = [
        str(ADV_PYTHON),
        str(PIPELINE_PY),
        "--resume-manifest",
        str(chunk_path),
        "--out-root",
        str(out_root),
        "--enable-demucs",
        "--enable-pyannote",
        "--enable-whisper",
        "--skip-voiceprint",
        "--demucs-batch-size",
        "1",
        "--pyannote-batch-size",
        "1",
        "--whisper-batch-size",
        "1",
        "--quality-thres",
        "0.65",
    ]
    log(f"[chunk] start {chunk_path.name} -> {out_root}")
    proc = subprocess.run(cmd, env=env)
    log(f"[chunk] done {chunk_path.name} rc={proc.returncode}")
    return int(proc.returncode)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_results(chunk_roots: list[Path]) -> None:
    selected: list[dict] = []
    for chunk_root in chunk_roots:
        report_path = chunk_root / "report.json"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        selected.extend(report.get("final_selected", []))

    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    final_wavs = MERGED_ROOT / "final" / "wavs"
    final_texts = MERGED_ROOT / "final" / "transcripts"
    train_wavs = MERGED_ROOT / "dataset" / "train" / "wavs"
    train_texts = MERGED_ROOT / "dataset" / "train" / "transcripts"
    val_wavs = MERGED_ROOT / "dataset" / "val" / "wavs"
    val_texts = MERGED_ROOT / "dataset" / "val" / "transcripts"
    for path in [final_wavs, final_texts, train_wavs, train_texts, val_wavs, val_texts]:
        path.mkdir(parents=True, exist_ok=True)

    ordered = list(selected)
    random.Random(42).shuffle(ordered)
    split_idx = int(len(ordered) * 0.9)
    train = ordered[:split_idx]
    val = ordered[split_idx:]

    for item in ordered:
        sample_id = item["sample_id"]
        found = None
        found_root = None
        for chunk_root in chunk_roots:
            candidate = chunk_root / "final" / "wavs" / f"{sample_id}.wav"
            if candidate.exists():
                found = candidate
                found_root = chunk_root
                break
        if found is None:
            continue
        shutil.copy2(found, final_wavs / found.name)
        txt_src = (found_root / "final" / "transcripts" / f"{sample_id}.txt") if found_root is not None else None
        if txt_src is not None and txt_src.exists():
            shutil.copy2(txt_src, final_texts / txt_src.name)

    for item in train:
        sample_id = item["sample_id"]
        wav_name = f"{sample_id}.wav"
        txt_name = f"{sample_id}.txt"
        src_wav = final_wavs / wav_name
        src_txt = final_texts / txt_name
        if src_wav.exists():
            shutil.copy2(src_wav, train_wavs / wav_name)
        if src_txt.exists():
            shutil.copy2(src_txt, train_texts / txt_name)
    for item in val:
        sample_id = item["sample_id"]
        wav_name = f"{sample_id}.wav"
        txt_name = f"{sample_id}.txt"
        src_wav = final_wavs / wav_name
        src_txt = final_texts / txt_name
        if src_wav.exists():
            shutil.copy2(src_wav, val_wavs / wav_name)
        if src_txt.exists():
            shutil.copy2(src_txt, val_texts / txt_name)

    report = {
        "selected": len(selected),
        "train": len(train),
        "val": len(val),
        "chunk_roots": [str(p) for p in chunk_roots],
    }
    (MERGED_ROOT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (MERGED_ROOT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in selected) + ("\n" if selected else ""),
        encoding="utf-8",
    )


def main() -> int:
    if not CHUNK_DIR.exists():
        log(f"[err] chunk dir missing: {CHUNK_DIR}")
        return 1
    chunk_files = sorted(CHUNK_DIR.glob("resume_chunk_*.jsonl"))
    if not chunk_files:
        log(f"[err] no chunk manifests found in {CHUNK_DIR}")
        return 1

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    chunk_roots: list[Path] = []
    for idx, chunk_path in enumerate(chunk_files, start=1):
        out_root = RUN_ROOT / f"chunk_{idx:02d}"
        out_root.mkdir(parents=True, exist_ok=True)
        chunk_roots.append(out_root)
        report_path = out_root / "report.json"
        if report_path.exists():
            log(f"[chunk] skip {chunk_path.name} (already completed)")
            continue
        rc = run_chunk(chunk_path, out_root)
        if rc != 0:
            log(f"[err] chunk failed: {chunk_path.name}")
            return rc

    merge_results(chunk_roots)
    log(f"[ok] merged output: {MERGED_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
