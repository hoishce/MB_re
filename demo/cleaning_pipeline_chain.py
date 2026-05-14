#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run core cleaning first, then continue in the advanced environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cleaning_pipeline_v2_impl import build_argument_parser, resolve_stage_flags


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PY = REPO_ROOT / "demo" / "cleaning_pipeline_v2.py"
ADVANCED_PYTHON = REPO_ROOT / ".venv_advanced" / "Scripts" / "python.exe"


def _arg_or_none(value):
    if value is None:
        return None
    return str(value)


def _add_option(argv: list[str], flag: str, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    argv.extend([flag, str(value)])


def namespace_to_argv(
    args: argparse.Namespace,
    *,
    out_root: str | None = None,
    resume_manifest: str | None = None,
    force_skip_advanced: bool = False,
    quality_thres_override: float | None = None,
) -> list[str]:
    argv: list[str] = []
    _add_option(argv, "--input-dir", args.input_dir)
    _add_option(argv, "--out-root", out_root if out_root is not None else args.out_root)
    _add_option(argv, "--run-version", args.run_version)
    _add_option(argv, "--dataset-root", args.dataset_root)
    _add_option(argv, "--speaker-ref", args.speaker_ref)
    _add_option(argv, "--batch-size", args.batch_size)
    _add_option(argv, "--voiceprint-batch-size", getattr(args, "voiceprint_batch_size", None))
    _add_option(argv, "--pyannote-batch-size", getattr(args, "pyannote_batch_size", None))
    _add_option(argv, "--whisper-batch-size", getattr(args, "whisper_batch_size", None))
    _add_option(argv, "--demucs-batch-size", getattr(args, "demucs_batch_size", None))
    _add_option(argv, "--dedupe-thresh", args.dedupe_thresh)
    _add_option(argv, "--quality-thres", quality_thres_override if quality_thres_override is not None else args.quality_thres)
    _add_option(argv, "--start-index", args.start_index)
    _add_option(argv, "--max-files", args.max_files)
    _add_option(argv, "--dry-run", args.dry_run)
    _add_option(argv, "--augment", args.augment)
    _add_option(argv, "--move", args.move)
    _add_option(argv, "--min-seg-sec", args.min_seg_sec)
    _add_option(argv, "--max-seg-sec", args.max_seg_sec)
    _add_option(argv, "--min-rms-db", args.min_rms_db)
    _add_option(argv, "--max-clip-ratio", args.max_clip_ratio)
    _add_option(argv, "--min-snr-db", args.min_snr_db)
    _add_option(argv, "--max-reverb-score", args.max_reverb_score)
    _add_option(argv, "--min-vad-ratio", args.min_vad_ratio)
    _add_option(argv, "--speaker-thres", args.speaker_thres)
    _add_option(argv, "--train-ratio", args.train_ratio)
    _add_option(argv, "--filter-multi-speaker", args.filter_multi_speaker)
    _add_option(argv, "--allow-multi-speaker", args.allow_multi_speaker)
    if resume_manifest is not None:
        _add_option(argv, "--resume-manifest", resume_manifest)

    if force_skip_advanced:
        argv.extend(["--skip-demucs", "--skip-whisper", "--skip-pyannote", "--skip-voiceprint"])
    else:
        _add_option(argv, "--skip-demucs", args.skip_demucs)
        _add_option(argv, "--skip-whisper", args.skip_whisper)
        _add_option(argv, "--skip-pyannote", args.skip_pyannote)
        _add_option(argv, "--skip-voiceprint", args.skip_voiceprint)
        _add_option(argv, "--enable-demucs", args.enable_demucs)
        _add_option(argv, "--enable-whisper", args.enable_whisper)
        _add_option(argv, "--enable-pyannote", args.enable_pyannote)
        _add_option(argv, "--enable-voiceprint", args.enable_voiceprint)

    return argv


def run_subprocess(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    proc = subprocess.run(cmd, env=env)
    return int(proc.returncode)


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    flags = resolve_stage_flags(args)

    has_advanced_request = any(
        [
            flags["demucs"],
            flags["whisper"],
            flags["pyannote"],
            flags["voiceprint"],
        ]
    )

    core_quality_thres = 0.0 if has_advanced_request else None
    core_cmd = [sys.executable, str(PIPELINE_PY)] + namespace_to_argv(
        args,
        force_skip_advanced=True,
        quality_thres_override=core_quality_thres,
    )
    core_rc = run_subprocess(core_cmd)
    if core_rc != 0:
        return core_rc

    if not has_advanced_request or args.dry_run:
        return 0

    resume_manifest = Path(args.out_root) / "resume_manifest.jsonl"
    if not resume_manifest.exists():
        print(f"resume manifest not found: {resume_manifest}", file=sys.stderr)
        return 1

    advanced_python = ADVANCED_PYTHON
    if not advanced_python.exists():
        print(
            f"advanced environment python not found: {advanced_python}\n"
            "Please create the advanced environment first.",
            file=sys.stderr,
        )
        return 1

    advanced_out_root = str(Path(args.out_root) / "advanced")
    advanced_cmd = [str(advanced_python), str(PIPELINE_PY)] + namespace_to_argv(
        args,
        out_root=advanced_out_root,
        resume_manifest=str(resume_manifest),
        force_skip_advanced=False,
    )
    advanced_env = os.environ.copy()
    advanced_env["MOCKINGBIRD_PIPELINE_PROFILE"] = "advanced"
    advanced_rc = run_subprocess(advanced_cmd, env=advanced_env)
    return advanced_rc


if __name__ == "__main__":
    raise SystemExit(main())
