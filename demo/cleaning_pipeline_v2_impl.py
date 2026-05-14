#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cleaning_pipeline_v2_impl.py

主清洗引擎，围绕以下流程组织：
  下载/导入原始音频
  -> 格式转换(16kHz/单声道/WAV)
  -> 切片(3~10 秒) + MFCC 指纹去重
  -> Pre-A 音量/削波检测
  -> Demucs(可选)
  -> Pre-B SNR/混响检测
  -> VAD(优先 Silero,回退到 librosa)
  -> pyannote 说话人分离(可选)
  -> 声纹比对(可选)
  -> Whisper 转录(可选)
  -> Post 质量打分
  -> 标准化数据集导出(train/val + manifest)

默认只启用核心清洗链路:Demucs / pyannote / Whisper / 声纹比对都通过
显式开关启用，便于把高级链路拆到单独环境。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Any, Callable

LOG = logging.getLogger("cleaning_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TARGET_NUMPY = "1.26.4"
ADVANCED_NUMPY_MIN = (2, 2, 2)
QUALITY_WEIGHTS = {
    "speaker": 0.35,
    "snr": 0.30,
    "vad": 0.20,
    "transcript": 0.15,
}
QUALITY_PENALTIES = {
    "reverb": 0.10,
    "clip": 0.05,
}
MIN_FINAL_DURATION_SEC = 5.0
MIN_FINAL_RMS_DB = -35.0
TARGET_SIM_TH = 0.72
NEGATIVE_MARGIN_TH = 0.10
MAX_MUSIC_SCORE = 0.25
MAX_SILENT_RATIO = 0.45
MIN_VAD_RATIO = 0.60
MIN_ASR_TEXT_LEN = 2
DEMUCS_MIN_RMS_DB = -35.0
DEMUCS_MAX_SILENT_RATIO = 0.40
DEMUCS_MIN_SPEECH_RATIO = 0.60
DEMUCS_MIN_ATTEMPT_SEC = 10.0
DEMUCS_MAX_ATTEMPT_SEC = 30.0
DEMUCS_UNCERTAIN_LOW_BAND_RATIO = 0.55
DEMUCS_UNCERTAIN_HIGH_BAND_RATIO = 0.28
WHISPER_MIN_TEXT_LEN = 4
WHISPER_MIN_CHINESE_CHARS = 1
WHISPER_MIN_CONFIDENCE = 0.35
WHISPER_MAX_NO_SPEECH_PROB = 0.55
WHISPER_MIN_DURATION_RATIO = 0.25
WHISPER_MAX_DURATION_RATIO = 1.50
SPEAKER_SIM_PASS = 0.68
SPEAKER_SIM_UNCERTAIN = 0.58
SPEAKER_SIM_REJECT = 0.50
HIGH_CONF_SPEAKER_SIM = 0.70
UNCERTAIN_SPEAKER_SIM = 0.55
VOICEPRINT_PASS_TH = TARGET_SIM_TH
VOICEPRINT_UNCERTAIN_TH = 0.58
MIN_FINAL_SPEAKER_SIM = TARGET_SIM_TH
FINAL_MIN_DURATION_SEC = 1.2
FINAL_MAX_DURATION_SEC = 15.0
FINAL_MIN_RMS_DB = -40.0
FINAL_MAX_SILENT_RATIO = MAX_SILENT_RATIO
FINAL_TARGET_TOTAL_SECONDS = 600.0
FINAL_MAX_FILES = 80
VAD_AGGRESSIVENESS = 1
MIN_SPEECH_DURATION = 0.6
MIN_SEGMENT_DURATION = 1.0
MIN_SPEECH_RATIO = 0.15
MAX_SILENCE_RATIO = 0.85
TRANSCRIPT_REJECT_PATTERNS = (
    "\u5b57\u5e55",
    "\u4e2d\u6587\u5b57\u5e55",
    "\u5b57\u5e55\u5fd7\u613f\u8005",
    "\u5fd7\u613f\u8005",
    "\u611f\u8c22\u89c2\u770b",
    "\u6b22\u8fce\u6536\u770b",
    "\u8bf7\u52ff\u8f6c\u8f7d",
    "\u72ec\u64ad\u5267\u573a",
    "\u4f18\u4f18\u72ec\u64ad\u5267\u573a",
    "yoyo",
    "television",
    "series",
    "exclusive",
    "by",
    "music",
    "\u6b4c\u8bcd",
    "\u6f14\u5531",
    "mv",
)
INTERVIEW_TITLE_HINTS = (
    "采访",
    "专访",
    "访谈",
    "对谈",
    "对话",
    "人物",
    "interview",
    "dialogue",
    "conversation",
)
VOICEPRINT_REVIEW_TITLE_HINTS = (
    "爆料",
    "热搜",
    "丑闻",
    "揭秘",
    "解说",
    "评论",
    "吐槽",
    "争议",
    "controversy",
    "explained",
)
DEMUCS_LEGACY_PYTHON = Path(__file__).resolve().parents[1] / ".venv_demucs_legacy" / "Scripts" / "python.exe"
DEMUCS_LEGACY_PYTHON_ENV = "MOCKINGBIRD_DEMUCS_PYTHON"


def resolve_demucs_python() -> str:
    override = os.environ.get(DEMUCS_LEGACY_PYTHON_ENV)
    if override:
        return override
    if DEMUCS_LEGACY_PYTHON.exists():
        return str(DEMUCS_LEGACY_PYTHON)
    return sys.executable


PYANNOTE_PIPELINE_CANDIDATES = [
    (
        os.environ.get("MOCKINGBIRD_PYANNOTE_PIPELINE", "pyannote/speaker-diarization-community-1"),
        os.environ.get("MOCKINGBIRD_PYANNOTE_REVISION", "main"),
    ),
    ("pyannote/speaker-diarization", "2.1"),
]


def runtime_profile() -> str:
    return os.environ.get("MOCKINGBIRD_PIPELINE_PROFILE", "core").strip().lower() or "core"


def parse_version_tuple(version_str: str) -> tuple[int, int, int]:
    head = version_str.split("+", 1)[0]
    parts = head.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = 0
    if len(parts) > 2:
        match = re.match(r"(\d+)", parts[2])
        if match:
            patch = int(match.group(1))
    return major, minor, patch


def require_numpy_compatibility() -> None:
    """Fail fast when the runtime NumPy is not the pinned release."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception as exc:  # pragma: no cover - stdlib import guard
        raise RuntimeError("Python importlib.metadata is unavailable") from exc

    try:
        current = version("numpy").split("+", 1)[0]
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "NumPy is required for the cleaning pipeline. "
            "Please install dependencies from requirements.txt."
        ) from exc

    profile = runtime_profile()
    if profile == "advanced":
        if parse_version_tuple(current) < ADVANCED_NUMPY_MIN:
            raise RuntimeError(
                f"Detected NumPy {current}, but the advanced pipeline expects NumPy "
                f"{ADVANCED_NUMPY_MIN[0]}.{ADVANCED_NUMPY_MIN[1]}.{ADVANCED_NUMPY_MIN[2]} or newer. "
                "Please run the advanced stages inside the advanced environment."
            )
    else:
        if current != TARGET_NUMPY:
            raise RuntimeError(
                f"Detected NumPy {current}, but this pipeline expects NumPy {TARGET_NUMPY}. "
                "Please reinstall dependencies and keep the pinned version."
            )


require_numpy_compatibility()


def load_cfg_from_pipeline() -> dict:
    """尝试读取现有 pipeline.py 中的 CFG。"""
    try:
        repo_root = Path(__file__).resolve().parents[1]
        pipeline_py = repo_root / "demo" / "pipeline.py"
        spec = importlib.util.spec_from_file_location("demo_pipeline_cfg", str(pipeline_py))
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "CFG", {}) or {}
    except Exception:
        return {}


CFG = load_cfg_from_pipeline()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str, maxlen: int = 64) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", text).strip("_")
    return cleaned[:maxlen] or "item"


def sha1_short(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def make_run_version(value: str | None) -> str:
    if value:
        return value
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def iter_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    size = max(1, int(batch_size or 1))
    return [items[i : i + size] for i in range(0, len(items), size)]


def cleanup_runtime_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def copy_or_move(src: Path, dst: Path, move: bool = False) -> Path:
    ensure_dir(dst.parent)
    if move:
        if src.resolve() != dst.resolve():
            if dst.exists():
                dst.unlink()
            src.replace(dst)
    else:
        shutil.copy2(src, dst)
    return dst


def path_key(p: str | Path) -> str:
    return os.path.normcase(Path(p).resolve().as_posix())


def stage_pcm16_path(cache_root: Path, src: Path, label: str) -> Path:
    stem = f"{safe_name(src.stem, 48)}__{sha1_short(path_key(src))}"
    return cache_root / label / f"{stem}_16k.wav"


def ensure_pcm16_wav(src: str | Path, dst: str | Path) -> tuple[bool, str]:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(dst_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not dst_path.exists():
        stderr = (proc.stderr or "").strip()
        return False, stderr[-1000:]
    return True, ""


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_audio_numpy(path: Path):
    """返回 waveform 和采样率。"""
    try:
        import soundfile as sf

        y, sr = sf.read(str(path))
        return y, sr
    except Exception as exc:
        raise RuntimeError(f"failed to read audio with soundfile: {exc}") from exc


def convert_to_wav16_mono(src: Path, dst: Path) -> None:
    """Convert to 16kHz / mono / PCM16 WAV using pydub."""
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError("pydub is required for format conversion") from exc

    audio = AudioSegment.from_file(str(src))
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    ensure_dir(dst.parent)
    audio.export(str(dst), format="wav")


def rms_db(y) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return -999.0
    arr = np.asarray(y)
    rms = math.sqrt(float((arr**2).mean()))
    return 20.0 * math.log10(max(rms, 1e-12))


def detect_clipping(y, threshold: float = 0.9995) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return 0.0
    arr = np.asarray(y)
    mx = max(abs(arr.max()), abs(arr.min())) if hasattr(arr, "max") else 1.0
    if mx == 0:
        return 0.0
    normalized = arr / float(mx)
    return float((abs(normalized) >= threshold).sum()) / float(len(normalized))


def estimate_snr_db(y, sr) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return -999.0
    arr = np.asarray(y)
    frame_len = max(1, int(0.02 * sr))
    energies = []
    for i in range(0, max(1, len(arr) - frame_len), frame_len):
        frame = arr[i : i + frame_len]
        if len(frame) == 0:
            continue
        energies.append(float((frame**2).mean()))
    if not energies:
        return -999.0
    energies = np.asarray(energies)
    noise = float(np.percentile(energies, 10))
    signal = float(np.percentile(energies, 90))
    if noise <= 0:
        return 60.0
    return 10.0 * math.log10(max(signal / noise, 1e-6))


def estimate_reverb_score(y) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return 0.0
    arr = np.asarray(y)
    n = len(arr)
    a = float(np.mean(arr[: max(1, n // 5)] ** 2)) + 1e-12
    b = float(np.mean(arr[-max(1, n // 3) :] ** 2)) + 1e-12
    return float(min(1.0, math.log1p(b / a)))


def peak_dbfs(y) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return -999.0
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = float(np.max(np.abs(arr)))
    return 20.0 * math.log10(max(peak, 1e-12))


def normalize_audio_file_to_dbfs(audio_path: Path, target_dbfs: float = -19.0) -> bool:
    try:
        from pydub import AudioSegment
    except Exception:
        return False
    try:
        audio = AudioSegment.from_file(str(audio_path))
        if len(audio) == 0:
            return False
        current = float(audio.dBFS)
        if current == float("-inf") or math.isnan(current):
            return False
        audio = audio.apply_gain(target_dbfs - current)
        audio.export(str(audio_path), format="wav")
        return True
    except Exception:
        return False


def silent_ratio(y, sr: int, frame_ms: int = 30, silence_dbfs: float = -40.0) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return 1.0
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    frame_len = max(1, int(sr * frame_ms / 1000))
    frame_count = len(arr) // frame_len
    if frame_count <= 0:
        return 1.0
    frames = arr[: frame_count * frame_len].reshape(frame_count, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_dbfs = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    return float(np.mean(frame_dbfs <= silence_dbfs))


def speech_ratio(y, sr: int) -> float:
    return max(0.0, 1.0 - silent_ratio(y, sr))


def spectral_band_profile(y, sr: int) -> dict[str, float]:
    import numpy as np

    if y is None or len(y) == 0:
        return {"low_ratio": 0.0, "high_ratio": 0.0, "centroid": 0.0}
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if not np.any(np.abs(arr)):
        return {"low_ratio": 1.0, "high_ratio": 0.0, "centroid": 0.0}
    raw_power = int(math.log2(max(2, len(arr)))) - 1
    n_fft = 1 << max(9, min(11, raw_power))
    spec = np.abs(np.fft.rfft(arr, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    total = float(spec.sum()) + 1e-12
    low_ratio = float(spec[freqs < 250.0].sum() / total)
    high_ratio = float(spec[freqs > 5500.0].sum() / total)
    centroid = float((freqs * spec).sum() / total) if total > 0 else 0.0
    return {"low_ratio": low_ratio, "high_ratio": high_ratio, "centroid": centroid}


def is_spectrum_uncertain(y, sr: int) -> bool:
    profile = spectral_band_profile(y, sr)
    return profile["low_ratio"] > DEMUCS_UNCERTAIN_LOW_BAND_RATIO or profile["high_ratio"] > DEMUCS_UNCERTAIN_HIGH_BAND_RATIO


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def whisper_text_quality(text: str, stats: dict[str, Any] | None = None) -> tuple[bool, bool, float, list[str]]:
    clean = normalize_match_text(text)
    reasons: list[str] = []
    if not clean:
        reasons.append("empty")
    if len(clean) < WHISPER_MIN_TEXT_LEN:
        reasons.append("too_short")
    if not contains_chinese(text):
        reasons.append("no_chinese")

    audio_duration = 0.0
    transcript_duration = 0.0
    avg_logprob = None
    no_speech_prob = None
    quality_conf = 0.0
    if stats:
        audio_duration = float(stats.get("audio_duration_sec", 0.0) or 0.0)
        transcript_duration = float(stats.get("transcript_duration_sec", 0.0) or 0.0)
        avg_logprob = stats.get("avg_logprob")
        no_speech_prob = stats.get("no_speech_prob")
        quality = stats.get("quality") or {}
        quality_conf = float(quality.get("confidence", 0.0) or 0.0)
        if no_speech_prob is not None and float(no_speech_prob) > WHISPER_MAX_NO_SPEECH_PROB:
            reasons.append("high_no_speech_prob")
        if avg_logprob is not None and float(avg_logprob) < -1.2:
            reasons.append("low_logprob")
        if audio_duration > 0.0:
            ratio = transcript_duration / max(audio_duration, 1e-6)
            if ratio < WHISPER_MIN_DURATION_RATIO or ratio > WHISPER_MAX_DURATION_RATIO:
                reasons.append("duration_mismatch")
        if quality_conf < WHISPER_MIN_CONFIDENCE:
            reasons.append("low_confidence")

    accepted = not reasons
    uncertain = accepted and (
        quality_conf < 0.55
        or len(clean) < 12
        or (no_speech_prob is not None and float(no_speech_prob) > 0.25)
    )
    return accepted, uncertain, quality_conf, reasons


def frame_silent_ratio(y, sr: int, frame_ms: int = 30, silence_dbfs: float = -40.0) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return 1.0
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    frame_len = max(1, int(sr * frame_ms / 1000))
    frame_count = len(arr) // frame_len
    if frame_count <= 0:
        return 1.0
    frames = arr[: frame_count * frame_len].reshape(frame_count, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_dbfs = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    return float(np.mean(frame_dbfs <= silence_dbfs))


def peak_abs(y) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return 0.0
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return float(np.max(np.abs(arr)))


def is_likely_music_like(y, sr: int) -> bool:
    import numpy as np
    import librosa

    if y is None or len(y) == 0:
        return True
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.sum(np.abs(arr)) <= 1e-12:
        return True
    harmonic, _ = librosa.effects.hpss(arr)
    harmonic_ratio = float(np.sum(np.abs(harmonic)) / (np.sum(np.abs(arr)) + 1e-10))
    flatness = float(librosa.feature.spectral_flatness(y=arr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(arr).mean())
    return harmonic_ratio > 0.45 and flatness < 0.35 and zcr < 0.12


def demucs_vocals_quality(y, sr: int) -> dict[str, Any]:
    rms = rms_db(y)
    return {
        "rms_db": rms,
        "peak": peak_abs(y),
        "silent_ratio": frame_silent_ratio(y, sr),
        "music_like": is_likely_music_like(y, sr),
    }


def compute_music_score(record: dict[str, Any]) -> float:
    metrics = record.get("metrics", {}) or {}
    score = 0.0
    if bool(metrics.get("music_like")) or bool(metrics.get("demucs_music_like")):
        score += 0.70
    if bool(metrics.get("spectrum_uncertain")):
        score += 0.15
    reverb_score = float(metrics.get("reverb_score", record.get("reverb_score", 0.0)) or 0.0)
    if reverb_score > 0.20:
        score += min(0.10, (reverb_score - 0.20) * 0.25)
    silent_ratio_value = float(metrics.get("silent_ratio", record.get("silent_ratio", 0.0)) or 0.0)
    if silent_ratio_value > 0.30:
        score += min(0.05, (silent_ratio_value - 0.30) * 0.10)
    vad_ratio = float(metrics.get("vad_ratio", 1.0) or 1.0)
    if vad_ratio < 0.60:
        score += 0.05
    return float(max(0.0, min(1.0, score)))


def build_stage_dirs(out_root: Path) -> dict[str, Path]:
    dirs = {
        "raw": out_root / "raw",
        "converted": out_root / "converted",
        "segments": out_root / "segments",
        "analysis": out_root / "analysis",
        "routes": out_root / "routes",
        "debug": out_root / "debug",
        "pcm16": out_root / "pcm16",
        "loudness": out_root / "loudness",
        "denoised": out_root / "denoised",
        "dereverb": out_root / "dereverb",
        "pre_a": out_root / "pre_a",
        "demucs": out_root / "demucs",
        "demucs_attempts": out_root / "demucs_attempts",
        "pre_b": out_root / "pre_b",
        "vad": out_root / "vad",
        "diarized": out_root / "diarized",
        "verified": out_root / "verified",
        "review_uncertain": out_root / "review_uncertain",
        "verified_pass_debug": out_root / "debug" / "verified_pass",
        "final_selected": out_root / "final_selected",
        "final_selected_wavs": out_root / "final_selected" / "wavs",
        "final_selected_texts": out_root / "final_selected" / "transcripts",
        "manual_transcribe": out_root / "manual_transcribe",
        "manual_transcribe_wavs": out_root / "manual_transcribe" / "wavs",
        "manual_transcribe_texts": out_root / "manual_transcribe" / "transcripts",
        "reject": out_root / "reject",
        "reject_wavs": out_root / "reject" / "wavs",
        "reject_texts": out_root / "reject" / "transcripts",
        "final_wavs": out_root / "final" / "wavs",
        "final_texts": out_root / "final" / "transcripts",
        "final": out_root / "final",
        "dataset": out_root / "dataset",
        "manifest": out_root / "manifest.jsonl",
        "report": out_root / "report.json",
        "refs": out_root / "refs",
        "aug": out_root / "aug",
        "bins": out_root / "bins",
    }
    for path in dirs.values():
        if path.suffix:
            continue
        ensure_dir(path)
    for sub in ("deleted_below_0_7", "bin_070_075", "bin_075_080", "passed_080"):
        ensure_dir(dirs["bins"] / sub)
    return dirs


def build_dataset_dirs(dataset_root: Path) -> dict[str, Path]:
    dirs = {
        "root": dataset_root,
        "train_wavs": dataset_root / "train" / "wavs",
        "train_texts": dataset_root / "train" / "transcripts",
        "val_wavs": dataset_root / "val" / "wavs",
        "val_texts": dataset_root / "val" / "transcripts",
        "manifest": dataset_root / "manifest.jsonl",
        "report": dataset_root / "report.json",
    }
    for key, path in dirs.items():
        if key not in {"manifest", "report"}:
            ensure_dir(path)
    return dirs


def estimate_lufs_db(y, sr: int) -> float:
    """Best-effort LUFS estimate, falling back to RMS when pyloudnorm is unavailable."""
    try:
        import numpy as np
        import pyloudnorm as pyln  # type: ignore

        arr = np.asarray(y, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(arr))
    except Exception:
        return rms_db(y)


def write_json_file(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def trim_audio_to_vad_span(src: Path, dst: Path) -> bool:
    """Trim leading/trailing silence around detected speech."""
    try:
        import librosa
        import soundfile as sf
    except Exception:
        return False

    try:
        y, sr = read_audio_numpy(src)
        arr = y
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=16000)
            sr = 16000
        intervals = collect_speech_intervals(src, arr, sr)
        intervals = merge_close_intervals(intervals, gap_sec=0.35)
        if not intervals:
            return False
        start = max(0.0, intervals[0][0] - 0.12)
        end = min(float(len(arr)) / float(sr), intervals[-1][1] + 0.12)
        s = max(0, int(round(start * sr)))
        e = min(len(arr), int(round(end * sr)))
        if e <= s:
            return False
        ensure_dir(dst.parent)
        sf.write(str(dst), arr[s:e], sr, subtype="PCM_16")
        return True
    except Exception:
        return False


def apply_light_dereverb(y, sr: int):
    """Heuristic dereverb pass. Falls back to the original waveform on failure."""
    try:
        import librosa
        import numpy as np
        from scipy.ndimage import gaussian_filter1d, uniform_filter1d

        arr = np.asarray(y, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if not np.any(np.abs(arr)):
            return arr
        stft = librosa.stft(arr, n_fft=1024, hop_length=200, win_length=800)
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        frame_energy = mag.mean(axis=0)
        smooth = uniform_filter1d(frame_energy, size=5, mode="nearest")
        decay = np.clip(frame_energy / (smooth + 1e-6), 0.65, 1.0)
        decay = gaussian_filter1d(decay, sigma=1.0)
        mag2 = mag * decay[None, :]
        restored = librosa.istft(mag2 * phase, hop_length=200, win_length=800, length=len(arr))
        return restored.astype(np.float32)
    except Exception:
        return y


class AudioLoader:
    """Standardizes input audio into 16 kHz mono WAV files."""

    @staticmethod
    def convert(src: Path, dst: Path) -> None:
        convert_to_wav16_mono(src, dst)

    @staticmethod
    def load(path: Path):
        return read_audio_numpy(path)


class QualityAnalyzer:
    """Computes quality metrics and derives routing issues."""

    low_volume_rms_db = -35.0
    low_volume_peak_dbfs = -12.0
    max_silent_ratio = 0.40
    min_speech_ratio = 0.60
    min_snr_db = 12.0
    max_reverb_score = 0.35

    def analyze(self, record: dict[str, Any]) -> dict[str, Any]:
        path = Path(record["current_path"])
        y, sr = read_audio_numpy(path)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            try:
                import librosa

                y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=16000)
                sr = 16000
            except Exception:
                pass

        duration_sec = float(len(y)) / float(sr or 16000)
        metrics = {
            "duration_sec": duration_sec,
            "rms_db": rms_db(y),
            "peak_dbfs": peak_dbfs(y),
            "lufs_db": estimate_lufs_db(y, sr),
            "silent_ratio": silent_ratio(y, sr),
            "speech_ratio": speech_ratio(y, sr),
            "snr_db": estimate_snr_db(y, sr),
            "reverb_score": estimate_reverb_score(y),
            "music_like": bool(is_likely_music_like(y, sr)),
            "spectrum_uncertain": bool(is_spectrum_uncertain(y, sr)),
        }
        issues: list[str] = []
        if metrics["duration_sec"] < 2.0 or metrics["duration_sec"] > 12.0:
            issues.append("duration")
        if metrics["rms_db"] < self.low_volume_rms_db or metrics["peak_dbfs"] < self.low_volume_peak_dbfs:
            issues.append("low_volume")
        if metrics["silent_ratio"] > self.max_silent_ratio or metrics["speech_ratio"] < self.min_speech_ratio:
            issues.append("silence")
        if metrics["music_like"] or metrics["spectrum_uncertain"]:
            issues.append("music")
        if metrics["snr_db"] < self.min_snr_db:
            issues.append("noise")
        if metrics["reverb_score"] > self.max_reverb_score:
            issues.append("reverb")

        hard_drop = "duration" in issues and duration_sec < 1.5
        if hard_drop or (metrics["silent_ratio"] > 0.85):
            decision = "reject"
        elif not issues:
            decision = "pass"
        elif issues == ["low_volume"]:
            decision = "repair"
        else:
            decision = "uncertain"
        score = 100.0
        if "duration" in issues:
            score -= 18.0
        if "low_volume" in issues:
            score -= 15.0
        if "silence" in issues:
            score -= 22.0
        if "music" in issues:
            score -= 15.0
        if "noise" in issues:
            score -= 15.0
        if "reverb" in issues:
            score -= 15.0
        if decision == "uncertain":
            score -= 8.0
        if decision == "reject":
            score = 0.0

        return {
            "path": str(path),
            "metrics": metrics,
            "issues": issues,
            "decision": decision,
            "repairable": decision in {"repair", "uncertain"},
            "score": float(max(0.0, min(100.0, score))),
            "normalized_path": str(path),
        }


class Router:
    """Maps analyzer issues to repair steps."""

    def build_plan(self, analysis: dict[str, Any], record: dict[str, Any]) -> list[str]:
        issues = set(analysis.get("issues") or [])
        plan: list[str] = []
        if "low_volume" in issues:
            plan.append("loudness_fix")
        if "silence" in issues:
            plan.append("vad_trim")
        if "music" in issues:
            plan.append("vocal_separator")
        if "noise" in issues:
            plan.append("speech_denoiser")
        if "reverb" in issues:
            plan.append("dereverb_module")
        if not plan:
            plan.append("speaker_verification")
        return plan


class LoudnessFixer:
    def __init__(self, out_dir: Path, target_dbfs: float = -19.0):
        self.out_dir = out_dir
        self.target_dbfs = float(target_dbfs)

    def apply(self, record: dict[str, Any]) -> bool:
        src = Path(record["current_path"])
        dst = self.out_dir / src.name
        copy_or_move(src, dst, move=False)
        ok = normalize_audio_file_to_dbfs(dst, target_dbfs=self.target_dbfs)
        if not ok:
            return False
        record["current_path"] = str(dst)
        record["stage_paths"]["loudness"] = str(dst)
        record["stage_status"]["loudness"] = "passed"
        return True


class VocalSeparator:
    def __init__(self, out_dir: Path, uncertain_dir: Path | None = None, device: str = "auto", enabled: bool = True):
        self.out_dir = out_dir
        self.uncertain_dir = uncertain_dir
        self.device = device
        self.enabled = enabled

    def apply(self, record: dict[str, Any]) -> bool:
        if not self.enabled:
            record["stage_status"]["vocal_separator"] = "skipped"
            return True
        src = Path(record["current_path"])
        separated = demucs_separate(src, self.out_dir, device=self.device)
        if separated is None:
            record["stage_status"]["vocal_separator"] = "fallback"
            record["note"] = "demucs_failed_no_vocals"
            return False
        dst = self.out_dir / f"{record['sample_id']}.wav"
        copy_or_move(separated, dst, move=False)
        record["current_path"] = str(dst)
        record["stage_paths"]["demucs"] = str(dst)
        record["stage_status"]["vocal_separator"] = "passed"
        try:
            y, sr = read_audio_numpy(dst)
            metrics = demucs_vocals_quality(y, sr)
            record["metrics"]["demucs_rms_db"] = metrics["rms_db"]
            record["metrics"]["demucs_peak"] = metrics["peak"]
            record["metrics"]["demucs_silent_ratio"] = metrics["silent_ratio"]
            record["metrics"]["demucs_music_like"] = bool(metrics["music_like"])
            if (
                metrics["rms_db"] < DEMUCS_MIN_RMS_DB
                or metrics["silent_ratio"] > DEMUCS_MAX_SILENT_RATIO
                or metrics["music_like"]
            ):
                record["stage_status"]["vocal_separator"] = "uncertain"
                if self.uncertain_dir is not None:
                    ensure_dir(self.uncertain_dir)
                    uncertain_dst = self.uncertain_dir / dst.name
                    copy_or_move(dst, uncertain_dst, move=False)
                    record["stage_paths"]["review_uncertain"] = str(uncertain_dst)
        except Exception as exc:
            LOG.debug("vocal separator quality check failed for %s: %s", dst, exc)
        return True


class SpeechDenoiser:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def apply(self, record: dict[str, Any]) -> bool:
        try:
            from utils import logmmse
            import numpy as np
            import soundfile as sf
        except Exception:
            record["stage_status"]["speech_denoiser"] = "unavailable"
            return False

        src = Path(record["current_path"])
        try:
            y, sr = read_audio_numpy(src)
            arr = np.asarray(y, dtype=np.float64)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if len(arr) < int(sr * 0.3):
                record["stage_status"]["speech_denoiser"] = "skipped_short"
                return False
            head = arr[: max(1, int(sr * 0.15))]
            tail = arr[-max(1, int(sr * 0.15)) :]
            noise_wav = np.concatenate([head, tail]) if len(tail) else head
            profile = logmmse.profile_noise(noise_wav, sr)
            cleaned = logmmse.denoise(arr, profile, eta=0.10)
            dst = self.out_dir / src.name
            ensure_dir(dst.parent)
            sf.write(str(dst), cleaned, sr, subtype="PCM_16")
            record["current_path"] = str(dst)
            record["stage_paths"]["denoised"] = str(dst)
            record["stage_status"]["speech_denoiser"] = "passed"
            return True
        except Exception as exc:
            record["stage_status"]["speech_denoiser"] = f"error:{exc}"
            record["note"] = str(exc)
            return False


class DereverbModule:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def apply(self, record: dict[str, Any]) -> bool:
        try:
            import soundfile as sf
        except Exception:
            record["stage_status"]["dereverb_module"] = "unavailable"
            return False

        src = Path(record["current_path"])
        try:
            y, sr = read_audio_numpy(src)
            cleaned = apply_light_dereverb(y, sr)
            dst = self.out_dir / src.name
            ensure_dir(dst.parent)
            sf.write(str(dst), cleaned, sr, subtype="PCM_16")
            record["current_path"] = str(dst)
            record["stage_paths"]["dereverb"] = str(dst)
            record["stage_status"]["dereverb_module"] = "passed"
            return True
        except Exception as exc:
            record["stage_status"]["dereverb_module"] = f"error:{exc}"
            record["note"] = str(exc)
            return False


class VADSegmenter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def apply(self, record: dict[str, Any]) -> bool:
        src = Path(record["current_path"])
        dst = self.out_dir / src.name
        ok = trim_audio_to_vad_span(src, dst)
        if not ok:
            record["stage_status"]["vad_trim"] = "skipped"
            return False
        record["current_path"] = str(dst)
        record["stage_paths"]["vad_trim"] = str(dst)
        record["stage_status"]["vad_trim"] = "passed"
        return True


class SpeakerVerifier:
    def verify(self, records: list[dict[str, Any]], out_dir: Path, speaker_ref: str | None, enabled: bool, speaker_thres: float, batch_size: int, model_path: str | None, move: bool) -> tuple[list[dict[str, Any]], Counter[str]]:
        return run_voiceprint_stage(
            records,
            out_dir,
            speaker_ref=speaker_ref,
            enabled=enabled,
            speaker_thres=speaker_thres,
            batch_size=batch_size,
            model_path=model_path,
            move=move,
        )


class WhisperValidator:
    def validate(self, records: list[dict[str, Any]], batch_size: int = 2) -> dict[str, str]:
        return transcribe_with_whisper(records, batch_size=batch_size)


class FinalScorer:
    def score(self, record: dict[str, Any]) -> float:
        return quality_score(record)


def run_modular_repair_pipeline(
    records: list[dict[str, Any]],
    dirs: dict[str, Path],
    args: argparse.Namespace,
    flags: dict[str, bool],
) -> list[dict[str, Any]]:
    analyzer = QualityAnalyzer()
    router = Router()
    loudness = LoudnessFixer(dirs["loudness"], target_dbfs=-19.0)
    vocal_separator = VocalSeparator(dirs["demucs"], dirs["review_uncertain"], device="auto", enabled=flags["demucs"])
    denoiser = SpeechDenoiser(dirs["denoised"])
    dereverb = DereverbModule(dirs["dereverb"])
    vad_segmenter = VADSegmenter(dirs["vad"])
    error_log_path = dirs["debug"] / "quality_analyzer_errors.jsonl"
    kept: list[dict[str, Any]] = []

    for record in records:
        record.setdefault("analysis", {})
        record.setdefault("route_plan", [])
        record.setdefault("analysis_status", "")
        record.setdefault("analysis_score", 0.0)
        record.setdefault("path_for_vad", record.get("current_path"))
        record.setdefault("final_score_penalty", 0.0)
        try:
            analysis = analyzer.analyze(record)
            record["analysis"] = analysis
            record["metrics"].update(analysis.get("metrics") or {})
            record["analysis_score"] = float(analysis.get("score", 0.0) or 0.0)
            plan = router.build_plan(analysis, record)
            record["route_plan"] = plan
            write_json_file({"analysis": analysis, "route_plan": plan}, dirs["analysis"] / f"{record['sample_id']}.json")
            write_json_file({"sample_id": record["sample_id"], "route_plan": plan}, dirs["routes"] / f"{record['sample_id']}.json")

            for _round in range(max(1, int(getattr(args, "repair_rounds", 2)))):
                applied = False
                if "loudness_fix" in plan:
                    applied = loudness.apply(record) or applied
                if "vad_trim" in plan:
                    applied = vad_segmenter.apply(record) or applied
                if "vocal_separator" in plan:
                    applied = vocal_separator.apply(record) or applied
                if "speech_denoiser" in plan:
                    applied = denoiser.apply(record) or applied
                if "dereverb_module" in plan:
                    applied = dereverb.apply(record) or applied

                if not applied:
                    break

                analysis = analyzer.analyze(record)
                record["analysis"] = analysis
                record["metrics"].update(analysis.get("metrics") or {})
                record["analysis_score"] = float(analysis.get("score", 0.0) or 0.0)
                plan = router.build_plan(analysis, record)
                record["route_plan"] = plan
                write_json_file({"analysis": analysis, "route_plan": plan}, dirs["analysis"] / f"{record['sample_id']}.json")
                write_json_file({"sample_id": record["sample_id"], "route_plan": plan}, dirs["routes"] / f"{record['sample_id']}.json")
                if analysis.get("decision") == "pass":
                    break

            decision = str(analysis.get("decision") or "")
            record["path_for_vad"] = str(record.get("current_path") or record.get("path") or "")
            if decision == "reject":
                record["status"] = "dropped_quality_analyzer"
                record["active"] = False
                record["analysis_status"] = "dropped_quality_analyzer"
                record["stage_status"]["quality_analyzer"] = "dropped"
                record["note"] = ",".join(analysis.get("issues") or []) or "quality_analyzer_reject"
                continue

            if decision == "uncertain":
                record["status"] = "analysis_uncertain"
                record["analysis_status"] = "analysis_uncertain"
                record["stage_status"]["quality_analyzer"] = "uncertain"
                record["final_score_penalty"] = float(record.get("final_score_penalty", 0.0)) + 8.0
                uncertain_dir = dirs["review_uncertain"]
                ensure_dir(uncertain_dir)
                dst = uncertain_dir / Path(record["current_path"]).name
                copy_or_move(Path(record["current_path"]), dst, move=False)
                record["stage_paths"]["review_uncertain"] = str(dst)
                record["path_for_vad"] = str(dst)
                record["route"] = "analysis_to_vad"
            else:
                record["status"] = "analysis_pass"
                record["analysis_status"] = "analysis_pass"
                record["stage_status"]["quality_analyzer"] = "passed"
                record["route"] = "analysis_to_vad"

            kept.append(record)
        except Exception as exc:
            record["status"] = "dropped_quality_analyzer_error"
            record["active"] = True
            record["analysis_status"] = "analysis_error_fallback"
            record["analysis_score"] = 0.0
            record["route"] = "analysis_error_fallback"
            record["path_for_vad"] = str(record.get("current_path") or record.get("path") or "")
            record["final_score_penalty"] = float(record.get("final_score_penalty", 0.0)) + 12.0
            record["stage_status"]["quality_analyzer"] = f"error:{exc}"
            record["note"] = str(exc)
            append_jsonl_row(error_log_path, {
                "sample_id": record.get("sample_id"),
                "path": str(record.get("current_path") or record.get("path") or ""),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            kept.append(record)
    return kept


def run_modular_pipeline(
    args: argparse.Namespace,
    flags: dict[str, bool],
    run_version: str,
    out_root: Path,
    dirs: dict[str, Path],
    dataset_root: Path,
) -> int:
    raw_src_dir = Path(args.input_dir or Path(CFG.get("temp_dir") or "./pipeline_temp") / "raw")
    resume_mode = bool(args.resume_manifest)
    all_files: list[Path] = []
    source_entries: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    active_records: list[dict[str, Any]] = []

    if resume_mode:
        resume_path = Path(args.resume_manifest)
        if not resume_path.exists():
            LOG.error("resume manifest does not exist: %s", resume_path)
            return 1
        active_records = load_manifest_records(resume_path)
        segment_records = list(active_records)
        all_files = [Path(record.get("source_path") or record.get("current_path")) for record in active_records if record.get("source_path") or record.get("current_path")]
        LOG.info("Loaded %d records from resume manifest %s", len(active_records), resume_path)
        if args.dry_run:
            LOG.info("DRY RUN: resume mode would continue from the loaded manifest through the modular pipeline.")
            return 0
    else:
        if not raw_src_dir.exists():
            LOG.error("输入目录不存在: %s", raw_src_dir)
            return 1

        all_files = split_source_files(raw_src_dir, start_index=args.start_index, max_files=args.max_files)
        LOG.info("Found %d raw files in %s (processing %d)", len(all_files), raw_src_dir, len(all_files))

        if args.dry_run:
            dry_run_modular_summary(all_files, args, flags)
            return 0

        for source_path in all_files:
            try:
                source_entries.append(make_source_entry(source_path, dirs["raw"], move=args.move))
            except Exception as exc:
                LOG.warning("copy/move source failed %s: %s", source_path, exc)

        LOG.info("Converted stage will process %d source files", len(source_entries))

        loader = AudioLoader()
        for entry in source_entries:
            try:
                converted_path = dirs["converted"] / f"{Path(entry['raw_path']).stem}_16k.wav"
                loader.convert(Path(entry["raw_path"]), converted_path)
                entry["converted_path"] = str(converted_path)
            except Exception as exc:
                LOG.warning("转换失败 %s: %s", entry["raw_path"], exc)
                entry["converted_path"] = None

        convertible = [entry for entry in source_entries if entry.get("converted_path")]
        LOG.info("Converted %d files", len(convertible))

        for entry in convertible:
            try:
                segment_records.extend(
                    segment_source_file(
                        entry,
                        dirs["segments"],
                        min_seg_sec=args.min_seg_sec,
                        max_seg_sec=args.max_seg_sec,
                    )
                )
            except Exception as exc:
                LOG.warning("切片失败 %s: %s", entry["converted_path"], exc)

        LOG.info("Created %d segments", len(segment_records))

        active_records = simple_dedupe_by_mfcc(segment_records, thresh=args.dedupe_thresh)
        LOG.info("After dedupe: %d segments", len(active_records))

    for record in segment_records:
        record["run_version"] = run_version

    LOG.info("Quality analyzer/router start: %d items", len(active_records))
    active_records = run_modular_repair_pipeline(active_records, dirs, args, flags)
    LOG.info("After modular repair: %d items", len(active_records))

    vad_stats: Counter[str] = Counter()
    LOG.info("VAD stage start: %d items, min_ratio=%.2f", len(active_records), float(args.min_vad_ratio))
    active_records = run_vad_stage(
        active_records,
        dirs["vad"],
        min_vad_ratio=float(args.min_vad_ratio),
        move=args.move,
        stats=vad_stats,
    )
    LOG.info("VAD pass: %d", len(active_records))
    LOG.info("VAD stats: %s", dict(vad_stats))
    if vad_stats.get("vad_input", 0) > 0 and vad_stats.get("vad_pass", 0) == 0:
        LOG.warning("VAD passed 0 items while inputs were available; threshold may still be too strict.")

    LOG.info("Pyannote stage start: %d items, batch=%d, device=%s", len(active_records), max(1, int(args.pyannote_batch_size)), "auto")
    active_records = run_pyannote_stage(
        active_records,
        dirs["diarized"],
        enabled=flags["pyannote"],
        filter_multi_speaker=flags["filter_multi_speaker"],
        batch_size=max(1, int(args.pyannote_batch_size)),
        device="auto",
    )
    LOG.info("Diarization stage items: %d", len(active_records))

    LOG.info("Voiceprint stage start: %d items, batch=%d", len(active_records), max(1, int(args.voiceprint_batch_size or args.batch_size)))
    active_records, voiceprint_stats = run_voiceprint_stage(
        active_records,
        dirs["verified"],
        speaker_ref=args.speaker_ref,
        enabled=flags["voiceprint"],
        speaker_thres=args.speaker_thres,
        batch_size=max(1, int(args.voiceprint_batch_size or args.batch_size)),
        model_path=CFG.get("mb_encoder"),
        move=args.move,
    )
    LOG.info("Verified speakers: %d", len(active_records))

    whisper_validator = WhisperValidator()
    if flags["whisper"]:
        LOG.info("Whisper stage start: %d items, batch=%d", len(active_records), max(1, int(args.whisper_batch_size)))
        transcripts = whisper_validator.validate(active_records, batch_size=max(1, int(args.whisper_batch_size)))
    else:
        transcripts = {str(record.get("sample_id") or Path(record["current_path"]).stem): "" for record in active_records}
        for record in active_records:
            record["stage_status"]["whisper"] = "skipped"

    final_scorer = FinalScorer()
    for record in active_records:
        sample_id = str(record.get("sample_id") or Path(record["current_path"]).stem)
        txt = transcripts.get(sample_id, "")
        record["transcript"] = txt
        record["transcript_conf"] = compute_transcript_confidence(txt, record.get("metrics")) if txt else 0.0
        if flags["whisper"]:
            record["stage_status"]["whisper"] = "passed"
        txt_path = dirs["final_texts"] / f"{sample_id}.txt"
        ensure_dir(txt_path.parent)
        txt_path.write_text(txt, encoding="utf-8")
        record["quality"] = final_scorer.score(record)

    final_candidates = len([r for r in active_records if r.get("active", True)])
    selected, manual_transcribe, final_reject_reasons = select_final_records_v2(
        active_records,
        max_files=max(1, int(getattr(args, "final_max_files", FINAL_MAX_FILES) or FINAL_MAX_FILES)),
        target_total_seconds=float(getattr(args, "final_target_total_seconds", FINAL_TARGET_TOTAL_SECONDS)),
        min_speaker_sim=float(getattr(args, "final_min_speaker_sim", MIN_FINAL_SPEAKER_SIM)),
    )
    LOG.info("Final selected count: %d", len(selected))
    LOG.info("Manual transcribe count: %d", len(manual_transcribe))
    LOG.info("final_reject_reasons:")
    for reason, count in final_reject_reasons.most_common():
        LOG.info("  %s: %d", reason, count)

    export_verified_pass_debug(active_records, selected, dirs["verified_pass_debug"])
    LOG.info("Verified-pass debug export written to %s", dirs["verified_pass_debug"])

    rejected_items = [record for record in active_records if str(record.get("final_bucket") or "") == "reject"]
    export_quality_buckets(selected, manual_transcribe, rejected_items, dirs, transcripts)
    LOG.info("Final buckets exported: final_selected=%d manual_transcribe=%d reject=%d", len(selected), len(manual_transcribe), len(rejected_items))

    resume_manifest_path = out_root / "resume_manifest.jsonl"
    write_jsonl(selected, resume_manifest_path)

    if args.augment and selected:
        run_augmentations(selected, dirs["aug"])

    train_stats = export_final_dataset(
        selected,
        dirs["final"],
        dataset_root,
        transcripts,
        train_ratio=args.train_ratio,
    )

    write_jsonl(segment_records, dirs["manifest"])
    try:
        write_jsonl(selected, dataset_root / "manifest.jsonl")
    except Exception as exc:
        LOG.warning("Failed to write dataset manifest: %s", exc)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "run_config": {
            "input_dir": str(raw_src_dir),
            "out_root": str(out_root),
            "dataset_root": str(dataset_root),
            "speaker_ref": args.speaker_ref,
            "run_version": run_version,
            "stage_flags": flags,
            "batch_sizes": {
                "demucs": max(1, int(args.demucs_batch_size)),
                "pyannote": max(1, int(args.pyannote_batch_size)),
                "voiceprint": max(1, int(args.voiceprint_batch_size or args.batch_size)),
                "whisper": max(1, int(args.whisper_batch_size)),
            },
            "repair_rounds": max(1, int(args.repair_rounds)),
            "quality_thres": args.quality_thres,
            "dedupe_thres": args.dedupe_thresh,
            "final_max_files": max(1, int(args.final_max_files or FINAL_MAX_FILES)),
            "final_target_total_seconds": float(args.final_target_total_seconds),
            "final_min_speaker_sim": float(args.final_min_speaker_sim),
            "quality_weights": QUALITY_WEIGHTS,
        },
        "counts": {
            "raw_total": len(all_files),
            "processed": len(source_entries),
            "converted": len([e for e in source_entries if e.get("converted_path")]),
            "segments": len(segment_records),
            "unique_segments": len([r for r in segment_records if r.get("status") != "dropped_duplicate"]),
            "pre_a_pass": len([r for r in segment_records if r.get("stage_status", {}).get("pre_a") == "passed"]),
            "analysis_pass": len([r for r in segment_records if r.get("analysis_status") == "analysis_pass"]),
            "analysis_uncertain": len([r for r in segment_records if r.get("analysis_status") == "analysis_uncertain"]),
            "analysis_error_fallback": len([r for r in segment_records if r.get("analysis_status") == "analysis_error_fallback"]),
            "analysis_reject": len([r for r in segment_records if r.get("status") == "dropped_quality_analyzer"]),
            "vad_input": int(vad_stats.get("vad_input", 0)),
            "vad_pass": int(vad_stats.get("vad_pass", 0)),
            "vad_reject_no_speech": int(vad_stats.get("vad_reject_no_speech", 0)),
            "vad_reject_too_short": int(vad_stats.get("vad_reject_too_short", 0)),
            "vad_reject_read_error": int(vad_stats.get("vad_reject_read_error", 0)),
            "diarized_pass": len([r for r in segment_records if r.get("stage_status", {}).get("pyannote") in ("passed", "skipped", "unavailable")]),
            "verified_input": len([r for r in segment_records if r.get("stage_status", {}).get("vad") == "passed"]),
            "voiceprint_input": int(voiceprint_stats.get("voiceprint_input", 0)),
            "voiceprint_cli_status": voiceprint_stats.get("voiceprint_cli_status", ""),
            "voiceprint_result_count": int(voiceprint_stats.get("voiceprint_result_count", 0)),
            "voiceprint_written": int(voiceprint_stats.get("voiceprint_written", 0)),
            "voiceprint_compute_error": int(voiceprint_stats.get("voiceprint_compute_error", 0)),
            "voiceprint_missing_result": int(voiceprint_stats.get("voiceprint_missing_result", 0)),
            "voiceprint_missing_similarity": int(voiceprint_stats.get("voiceprint_missing_similarity", 0)),
            "verified_pass": int(voiceprint_stats.get("verified_pass", 0)),
            "verified_uncertain": int(voiceprint_stats.get("verified_uncertain", 0)),
            "verified_reject": int(voiceprint_stats.get("verified_reject", 0)),
            "final_candidates": final_candidates,
            "final_selected": len(selected),
            "train": train_stats.get("train", 0),
            "val": train_stats.get("val", 0),
        },
        "status_breakdown": summarize_records(segment_records),
        "output": {
            "manifest": str(dirs["manifest"]),
            "report": str(dirs["report"]),
            "final_wavs": str(dirs["final_wavs"]),
            "final_texts": str(dirs["final_texts"]),
            "dataset_root": str(dataset_root),
        },
        "final_reject_reasons": dict(final_reject_reasons),
        "final_selected": [
            {
                "sample_id": record["sample_id"],
                "path": record["current_path"],
                "quality": record.get("quality", 0.0),
                "final_score": record.get("final_score", 0.0),
                "final_route": record.get("final_route"),
                "analysis_status": record.get("analysis_status"),
                "analysis_score": record.get("analysis_score", 0.0),
                "speaker_similarity": record.get("speaker_similarity"),
                "snr_db": record["metrics"].get("snr_db"),
                "vad_ratio": record["metrics"].get("vad_ratio"),
                "transcript_conf": record.get("transcript_conf", 0.0),
                "transcript": record.get("transcript", ""),
                "run_version": record.get("run_version", run_version),
            }
            for record in selected
        ],
    }
    write_json_file(report, dirs["report"])
    LOG.info("Pipeline complete. Report written to %s", dirs["report"])
    return 0


def split_source_files(raw_src_dir: Path, start_index: int = 0, max_files: int | None = None) -> list[Path]:
    files = sorted(
        [
            p
            for p in raw_src_dir.rglob("*")
            if p.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
        ]
    )
    start = max(0, int(start_index or 0))
    if max_files is not None:
        return files[start : start + max_files]
    return files[start:]


def load_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("stage_paths", {})
            record.setdefault("stage_status", {})
            record.setdefault("metrics", {})
            record.setdefault("active", True)
            record.setdefault("status", "resume_loaded")
            record.setdefault("transcript", "")
            record.setdefault("transcript_conf", 0.0)
            record.setdefault("speaker_similarity", None)
            record.setdefault("quality", 0.0)
            record.setdefault("note", "")
            record.setdefault("run_version", "")
            record.setdefault("analysis", {})
            record.setdefault("route_plan", [])
            if "current_path" not in record:
                record["current_path"] = record.get("path") or record.get("segment_path")
            records.append(record)
    return records


def make_source_entry(source_path: Path, raw_dir: Path, move: bool = False) -> dict[str, Any]:
    raw_path = raw_dir / source_path.name
    copy_or_move(source_path, raw_path, move=move)
    source_key = sha1_short(str(source_path.resolve()))
    source_tag = safe_name(source_path.stem, 32)
    return {
        "source_name": source_path.name,
        "source_path": str(source_path),
        "source_key": source_key,
        "source_tag": source_tag,
        "raw_path": str(raw_path),
        "converted_path": None,
    }


def create_segment_record(
    source: dict[str, Any],
    segment_path: Path,
    segment_index: int,
    start_sample: int,
    end_sample: int,
    sr: int,
) -> dict[str, Any]:
    sample_id = f"{source['source_tag']}__{source['source_key']}__seg{segment_index:04d}"
    segment_id = f"{source['source_key']}_seg_{segment_index:04d}"
    duration = (end_sample - start_sample) / float(sr)
    return {
        "sample_id": sample_id,
        "segment_id": segment_id,
        "source_name": source["source_name"],
        "source_path": source["source_path"],
        "source_key": source["source_key"],
        "raw_path": source["raw_path"],
        "converted_path": source["converted_path"],
        "segment_path": str(segment_path),
        "current_path": str(segment_path),
        "segment_index": segment_index,
        "segment_start_sec": round(start_sample / float(sr), 3),
        "segment_end_sec": round(end_sample / float(sr), 3),
        "duration_sec": round(duration, 3),
        "fingerprint": "",
        "stage_paths": {
            "raw": source["raw_path"],
            "converted": source["converted_path"],
            "segment": str(segment_path),
        },
        "stage_status": {
            "segment": "passed",
        },
        "metrics": {},
        "transcript": "",
        "transcript_conf": 0.0,
        "speaker_similarity": None,
        "quality": 0.0,
        "status": "segment_created",
        "active": True,
        "split": "",
        "note": "",
    }


def segment_source_file(
    source: dict[str, Any],
    segments_dir: Path,
    min_seg_sec: float,
    max_seg_sec: float,
) -> list[dict[str, Any]]:
    import librosa

    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile is required for segment export") from exc

    y, sr = sf.read(str(source["converted_path"]))
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    intervals = collect_speech_intervals(Path(source["converted_path"]), y, sr)
    if len(intervals) == 0:
        intervals = [(0.0, float(len(y)) / float(sr))]
    intervals = merge_close_intervals(intervals, gap_sec=0.35)
    windows = build_speech_windows(intervals, total_sec=float(len(y)) / float(sr), min_seg_sec=min_seg_sec, max_seg_sec=max_seg_sec)
    if len(windows) == 0:
        windows = [(0.0, float(len(y)) / float(sr))]

    seg_records: list[dict[str, Any]] = []
    idx = 1
    source_dir = segments_dir / f"{source['source_tag']}__{source['source_key']}"
    ensure_dir(source_dir)

    for start_sec, end_sec in windows:
        start = max(0, int(round(start_sec * sr)))
        end = min(len(y), int(round(end_sec * sr)))
        dur = (end - start) / float(sr)
        if dur < min_seg_sec * 0.75:
            continue
        seg_path = source_dir / f"{source['source_tag']}__{source['source_key']}__seg{idx:04d}.wav"
        sf.write(str(seg_path), y[start:end], sr, subtype="PCM_16")
        seg_records.append(create_segment_record(source, seg_path, idx, start, end, sr))
        idx += 1

    return seg_records


def collect_speech_intervals(path: Path, y=None, sr: int | None = None) -> list[tuple[float, float]]:
    """Prefer VAD timestamps so long files are cut around actual speech instead of silence."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio  # type: ignore

        model = load_silero_vad()
        wav = read_audio(str(path))
        speech = get_speech_timestamps(wav, model, return_seconds=True)
        intervals: list[tuple[float, float]] = []
        for seg in speech:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if end > start:
                intervals.append((start, end))
        if intervals:
            return intervals
    except Exception:
        pass

    try:
        import librosa
        import soundfile as sf

        if y is None or sr is None:
            y, sr = sf.read(str(path))
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)
            if sr != 16000:
                y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=16000)
                sr = 16000
        intervals_arr = librosa.effects.split(y, top_db=30)
        return [(float(s) / float(sr), float(e) / float(sr)) for s, e in intervals_arr if e > s]
    except Exception:
        return []


def merge_close_intervals(intervals: list[tuple[float, float]], gap_sec: float = 0.35) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    merged: list[list[float]] = [[sorted_intervals[0][0], sorted_intervals[0][1]]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap_sec:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return [(float(s), float(e)) for s, e in merged]


def build_speech_windows(
    intervals: list[tuple[float, float]],
    total_sec: float,
    min_seg_sec: float,
    max_seg_sec: float,
    pad_sec: float = 0.18,
) -> list[tuple[float, float]]:
    if not intervals:
        return []

    windows: list[tuple[float, float]] = []
    i = 0
    n = len(intervals)
    while i < n:
        start = intervals[i][0]
        end = intervals[i][1]
        j = i + 1
        while j < n:
            next_start, next_end = intervals[j]
            candidate_start = max(0.0, start - pad_sec)
            candidate_end = min(total_sec, next_end + pad_sec)
            if candidate_end - candidate_start <= max_seg_sec:
                end = next_end
                j += 1
                continue
            break

        candidate_start = max(0.0, start - pad_sec)
        candidate_end = min(total_sec, end + pad_sec)
        if candidate_end - candidate_start < min_seg_sec:
            target = min_seg_sec
            center = (start + end) / 2.0
            candidate_start = max(0.0, center - target / 2.0)
            candidate_end = min(total_sec, candidate_start + target)
            if candidate_end - candidate_start < target and candidate_start > 0.0:
                candidate_start = max(0.0, candidate_end - target)
        if candidate_end - candidate_start > max_seg_sec:
            candidate_end = candidate_start + max_seg_sec
        if candidate_end > candidate_start:
            windows.append((candidate_start, candidate_end))
        i = j

    # Fall back to gentle sliding windows if a very long speech run survived the grouping.
    final_windows: list[tuple[float, float]] = []
    for start, end in windows:
        if end - start <= max_seg_sec:
            final_windows.append((start, end))
            continue
        step = max(min_seg_sec, max_seg_sec - 0.5)
        cursor = start
        while cursor < end:
            seg_end = min(cursor + max_seg_sec, end)
            if seg_end - cursor >= min_seg_sec:
                final_windows.append((cursor, seg_end))
            cursor += step
    return final_windows


def compute_mfcc_fingerprint(path: Path) -> tuple[str, Any]:
    import librosa
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(path))
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    vec = np.mean(mfcc, axis=1)
    fingerprint = hashlib.sha1(np.round(vec, 3).tobytes()).hexdigest()[:16]
    return fingerprint, vec


def simple_dedupe_by_mfcc(records: list[dict[str, Any]], thresh: float = 0.985) -> list[dict[str, Any]]:
    import numpy as np
    from numpy.linalg import norm

    kept: list[dict[str, Any]] = []
    kept_vecs: list[Any] = []
    for record in records:
        try:
            fingerprint, vec = compute_mfcc_fingerprint(Path(record["current_path"]))
            record["fingerprint"] = fingerprint
            record["metrics"]["fingerprint"] = fingerprint
        except Exception as exc:
            record["stage_status"]["dedupe"] = f"fingerprint_failed:{exc}"
            kept.append(record)
            kept_vecs.append(None)
            continue

        duplicate = False
        for kept_vec in kept_vecs:
            if kept_vec is None:
                continue
            sim = float((vec @ kept_vec) / (norm(vec) * norm(kept_vec) + 1e-10))
            if sim >= thresh:
                duplicate = True
                break

        if duplicate:
            record["status"] = "dropped_duplicate"
            record["active"] = False
            record["stage_status"]["dedupe"] = "dropped"
            record["note"] = f"mfcc_similarity>={thresh}"
        else:
            record["stage_status"]["dedupe"] = "passed"
            kept.append(record)
            kept_vecs.append(vec)

    return kept


def demucs_separate_with_diagnostics(
    src: Path,
    out_dir: Path,
    device: str = "auto",
    timeout: int = 600,
) -> tuple[Path | None, dict[str, Any]]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    env = os.environ.copy()
    demucs_python = resolve_demucs_python()
    cmd = [demucs_python, "-m", "demucs.separate", "--two-stems=vocals", "--out", str(out_dir)]
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd += ["--device", "cpu"]
    elif device and device != "auto":
        cmd += ["--device", device]
    else:
        try:
            import torch

            if torch.cuda.is_available():
                cmd += ["--device", "cuda"]
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
                cmd += ["--device", "cpu"]
        except Exception:
                env["CUDA_VISIBLE_DEVICES"] = ""
                cmd += ["--device", "cpu"]
    cmd.append(str(src))
    diagnostics: dict[str, Any] = {
        "cmd": cmd,
        "input_path": str(src),
        "output_dir": str(out_dir),
        "device": device,
        "timeout": int(timeout),
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
        diagnostics["return_code"] = int(proc.returncode)
        diagnostics["stdout"] = proc.stdout or ""
        diagnostics["stderr"] = proc.stderr or ""
    except Exception as exc:
        LOG.warning("demucs subprocess failed: %s", exc)
        diagnostics["return_code"] = None
        diagnostics["stdout"] = ""
        diagnostics["stderr"] = str(exc)
        diagnostics["exception"] = repr(exc)
        proc = None

    candidates = list(out_dir.rglob("*vocals*.wav")) or list(out_dir.rglob("vocals.wav"))
    if not candidates:
        candidates = list(out_dir.rglob(f"{src.stem}*vocals*.wav"))
    if not candidates:
        if proc is not None and proc.returncode != 0:
            LOG.warning("demucs exited with code %s and produced no vocals; stderr=%s", proc.returncode, (proc.stderr or "")[:200])
            diagnostics["error"] = "no_vocals"
        else:
            diagnostics["error"] = "no_vocals"
        return None, diagnostics
    vocals = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    if proc is not None and proc.returncode != 0:
        LOG.info("demucs exited with code %s but produced vocals; treating as success", proc.returncode)
        diagnostics["warning"] = "nonzero_return_with_vocals"
    diagnostics["vocals_path"] = str(vocals)
    diagnostics["success"] = True
    return vocals, diagnostics


def demucs_separate(
    src: Path,
    out_dir: Path,
    device: str = "auto",
    timeout: int = 600,
) -> Path | None:
    vocals, _ = demucs_separate_with_diagnostics(src, out_dir, device=device, timeout=timeout)
    return vocals


def run_pre_a(records: list[dict[str, Any]], out_dir: Path, min_rms_db: float, max_clip_ratio: float, move: bool) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        try:
            y, sr = read_audio_numpy(Path(record["current_path"]))
            db = rms_db(y)
            clip = detect_clipping(y)
            record["metrics"]["rms_db"] = db
            record["metrics"]["clipping_ratio"] = clip
            if db < min_rms_db or clip > max_clip_ratio:
                record["status"] = "dropped_pre_a"
                record["active"] = False
                record["stage_status"]["pre_a"] = "dropped"
                record["note"] = f"rms_db={db:.2f}, clipping_ratio={clip:.4f}"
                continue
            dst = out_dir / Path(record["current_path"]).name
            copy_or_move(Path(record["current_path"]), dst, move=move)
            record["current_path"] = str(dst)
            record["stage_paths"]["pre_a"] = str(dst)
            record["stage_status"]["pre_a"] = "passed"
            kept.append(record)
        except Exception as exc:
            record["status"] = "dropped_pre_a_error"
            record["active"] = False
            record["stage_status"]["pre_a"] = f"error:{exc}"
            record["note"] = str(exc)
    return kept


def run_demucs_stage(
    records: list[dict[str, Any]],
    out_dir: Path,
    enabled: bool,
    device: str = "auto",
    batch_size: int = 1,
    attempts_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if not enabled:
        for record in records:
            record["stage_status"]["demucs"] = "skipped"
            record["stage_status"]["demucs_quality"] = record["stage_status"].get("demucs_quality") or "fallback_original"
        return records

    kept: list[dict[str, Any]] = []
    for batch in iter_batches(records, batch_size):
        for record in batch:
            src = Path(record["current_path"])
            record["stage_paths"]["demucs_input"] = str(src)
            duration_sec = float(record.get("duration_sec", 0.0) or record.get("metrics", {}).get("duration_sec", 0.0) or 0.0)
            if duration_sec < DEMUCS_MIN_ATTEMPT_SEC or duration_sec > DEMUCS_MAX_ATTEMPT_SEC:
                record["stage_status"]["demucs"] = "skipped_duration"
                record["stage_status"]["demucs_quality"] = "fallback_original"
                record["metrics"]["demucs_reason"] = "duration_out_of_range"
                kept.append(record)
                continue

            vocals, diagnostics = demucs_separate_with_diagnostics(src, out_dir, device=device)
            record["metrics"]["demucs_diagnostics"] = diagnostics
            if attempts_dir is not None:
                try:
                    ensure_dir(attempts_dir)
                    diag_path = attempts_dir / f"{record['sample_id']}.json"
                    write_json_file(diagnostics, diag_path)
                    record["stage_paths"]["demucs_attempt"] = str(diag_path)
                except Exception as exc:
                    LOG.debug("failed to write demucs diagnostics for %s: %s", src, exc)

            if vocals is not None:
                demucs_name = f"{record['sample_id']}.wav"
                demucs_dst = out_dir / demucs_name
                copy_or_move(vocals, demucs_dst, move=False)
                record["current_path"] = str(demucs_dst)
                record["stage_paths"]["demucs"] = str(demucs_dst)
                record["stage_status"]["demucs"] = "passed"
                record["metrics"]["demucs_used_source"] = "vocals"
            else:
                record["stage_status"]["demucs"] = "failed"
                record["stage_status"]["demucs_quality"] = "fallback_original"
                record["metrics"]["demucs_used_source"] = "original"
            kept.append(record)
        cleanup_runtime_cache()
    return kept


def run_demucs_quality_gate(
    records: list[dict[str, Any]],
    enabled_whisper_probe: bool,
    whisper_batch_size: int = 1,
    uncertain_dir: Path | None = None,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        try:
            current_path = Path(record["current_path"])
            demucs_input = Path(
                record.get("stage_paths", {}).get("demucs_input")
                or record.get("stage_paths", {}).get("pre_a")
                or record.get("stage_paths", {}).get("segment")
                or current_path
            )
            demucs_stage = str(record.get("stage_status", {}).get("demucs") or "")

            y, sr = read_audio_numpy(current_path)
            metrics = demucs_vocals_quality(y, sr)
            if metrics["rms_db"] < DEMUCS_MIN_RMS_DB and metrics["silent_ratio"] < 0.80:
                if normalize_audio_file_to_dbfs(current_path, target_dbfs=-19.0):
                    y, sr = read_audio_numpy(current_path)
                    metrics = demucs_vocals_quality(y, sr)

            used_source = "vocals"
            quality_state = "passed" if demucs_stage == "passed" else "fallback_original"
            spectrum_uncertain = is_spectrum_uncertain(y, sr)

            if demucs_stage == "passed" and (metrics["music_like"] or spectrum_uncertain or metrics["silent_ratio"] > DEMUCS_MAX_SILENT_RATIO):
                if demucs_input.exists() and demucs_input != current_path:
                    record["current_path"] = str(demucs_input)
                    record["stage_paths"]["demucs_fallback"] = str(demucs_input)
                    y, sr = read_audio_numpy(demucs_input)
                    metrics = demucs_vocals_quality(y, sr)
                    if metrics["rms_db"] < DEMUCS_MIN_RMS_DB and metrics["silent_ratio"] < 0.80:
                        if normalize_audio_file_to_dbfs(demucs_input, target_dbfs=-19.0):
                            y, sr = read_audio_numpy(demucs_input)
                            metrics = demucs_vocals_quality(y, sr)
                    used_source = "original"
                    quality_state = "fallback_original"
                    spectrum_uncertain = is_spectrum_uncertain(y, sr)
                else:
                    quality_state = "uncertain"

            hard_reject = (
                metrics["rms_db"] < (DEMUCS_MIN_RMS_DB - 8.0)
                or metrics["peak"] < 0.05
                or metrics["silent_ratio"] > 0.90
            )
            if hard_reject:
                if used_source == "vocals" and demucs_input.exists() and demucs_input != current_path:
                    record["current_path"] = str(demucs_input)
                    record["stage_paths"]["demucs_fallback"] = str(demucs_input)
                    y, sr = read_audio_numpy(demucs_input)
                    metrics = demucs_vocals_quality(y, sr)
                    used_source = "original"
                    quality_state = "fallback_original"
                    spectrum_uncertain = is_spectrum_uncertain(y, sr)
                    hard_reject = (
                        metrics["rms_db"] < (DEMUCS_MIN_RMS_DB - 8.0)
                        or metrics["peak"] < 0.05
                        or metrics["silent_ratio"] > 0.90
                    )
                if hard_reject:
                    record["status"] = "dropped_demucs_quality"
                    record["active"] = False
                    record["stage_status"]["demucs_quality"] = "rejected"
                    record["note"] = (
                        f"demucs_quality rms={metrics['rms_db']:.2f}, peak={metrics['peak']:.4f}, "
                        f"silent_ratio={metrics['silent_ratio']:.3f}, music_like={metrics['music_like']}"
                    )
                    continue

            if spectrum_uncertain or metrics["music_like"]:
                if quality_state != "fallback_original":
                    quality_state = "uncertain"
                record["note"] = "demucs_spectrum_uncertain" if spectrum_uncertain else "demucs_music_like"
                if uncertain_dir is not None:
                    try:
                        ensure_dir(uncertain_dir)
                        dst = uncertain_dir / f"{record.get('sample_id') or Path(record['current_path']).stem}_{Path(record['current_path']).name}"
                        copy_or_move(Path(record["current_path"]), dst, move=False)
                        record.setdefault("stage_paths", {})["review_uncertain"] = str(dst)
                    except Exception as exc:
                        LOG.debug("failed to archive uncertain demucs sample %s: %s", record.get("current_path"), exc)

            record["stage_status"]["demucs_quality"] = quality_state
            record["metrics"]["demucs_rms_db"] = metrics["rms_db"]
            record["metrics"]["demucs_peak"] = metrics["peak"]
            record["metrics"]["demucs_silent_ratio"] = metrics["silent_ratio"]
            record["metrics"]["demucs_music_like"] = bool(metrics["music_like"])
            record["metrics"]["demucs_used_source"] = used_source

            if enabled_whisper_probe:
                try:
                    probe_texts = transcribe_with_whisper([record], batch_size=1)
                except Exception as exc:
                    LOG.warning("demucs whisper probe unavailable: %s", exc)
                    probe_texts = {}
                sample_id = str(record.get("sample_id") or Path(record["current_path"]).stem)
                text = str(probe_texts.get(sample_id, "") or "")
                record["metrics"]["demucs_whisper_probe"] = text
                quality = record.get("metrics", {}).get("whisper_quality") or {}
                accepted, uncertain, conf, reasons = whisper_text_quality(text, quality if isinstance(quality, dict) else None)
                record["metrics"]["demucs_whisper_probe_conf"] = conf
                record["metrics"]["demucs_whisper_probe_reasons"] = reasons
                if not accepted:
                    if quality_state == "passed":
                        record["stage_status"]["demucs_quality"] = "uncertain"
                    record["note"] = "whisper_unrecognized_after_demucs"
                elif uncertain:
                    if uncertain_dir is not None:
                        try:
                            ensure_dir(uncertain_dir)
                            dst = uncertain_dir / f"{sample_id}_{Path(record['current_path']).name}"
                            if not dst.exists():
                                copy_or_move(Path(record["current_path"]), dst, move=False)
                            record.setdefault("stage_paths", {})["review_uncertain"] = str(dst)
                        except Exception as exc:
                            LOG.debug("failed to archive uncertain whisper sample %s: %s", record.get("current_path"), exc)
                    if quality_state == "passed":
                        record["stage_status"]["demucs_quality"] = "uncertain"
                record["transcript"] = text
                record["transcript_conf"] = compute_transcript_confidence(text, record.get("metrics"))

            kept.append(record)
        except Exception as exc:
            record["status"] = "dropped_demucs_quality_error"
            record["active"] = False
            record["stage_status"]["demucs_quality"] = f"error:{exc}"
            record["note"] = str(exc)

    return kept

def run_pre_b(records: list[dict[str, Any]], out_dir: Path, min_snr_db: float, max_reverb_score: float, move: bool) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        try:
            current_path = Path(record["current_path"])
            y, sr = read_audio_numpy(current_path)
            snr_db = estimate_snr_db(y, sr)
            reverb = estimate_reverb_score(y)
            rms = rms_db(y)
            silent = silent_ratio(y, sr)
            duration = float(record.get("duration_sec", 0.0) or len(y) / float(sr or 16000) if sr else 0.0)
            demucs_quality = str(record.get("stage_status", {}).get("demucs_quality") or "")

            record["metrics"]["snr_db"] = snr_db
            record["metrics"]["reverb_score"] = reverb
            record["metrics"]["pre_b_rms_db"] = rms
            record["metrics"]["pre_b_silent_ratio"] = silent

            catastrophic = (
                duration < 2.0
                or rms < (MIN_FINAL_RMS_DB - 8.0)
                or silent > 0.92
                or snr_db < (min_snr_db - 12.0)
            )
            if catastrophic:
                record["status"] = "dropped_pre_b"
                record["active"] = False
                record["stage_status"]["pre_b"] = "rejected"
                record["note"] = f"snr_db={snr_db:.2f}, reverb_score={reverb:.3f}, rms_db={rms:.2f}, silent_ratio={silent:.3f}"
                continue

            if demucs_quality == "passed" and snr_db >= min_snr_db and reverb <= max_reverb_score:
                route = "pass_vocals"
            elif demucs_quality in {"fallback_original", "failed", "skipped_duration"} and snr_db >= (min_snr_db - 2.0) and reverb <= min(0.99, max_reverb_score + 0.12):
                route = "fallback_original"
            else:
                route = "uncertain"

            dst = out_dir / Path(record["current_path"]).name
            copy_or_move(Path(record["current_path"]), dst, move=move)
            record["current_path"] = str(dst)
            record["stage_paths"]["pre_b"] = str(dst)
            record["stage_status"]["pre_b"] = route
            record["metrics"]["pre_b_route"] = route
            kept.append(record)
        except Exception as exc:
            record["status"] = "dropped_pre_b_error"
            record["active"] = False
            record["stage_status"]["pre_b"] = f"error:{exc}"
            record["note"] = str(exc)
    return kept

def build_vad_ratio_fn() -> tuple[Callable[[Path], float], str]:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio  # type: ignore

        model = load_silero_vad()

        def silero_ratio(path: Path) -> float:
            wav = read_audio(str(path))
            speech = get_speech_timestamps(wav, model, return_seconds=True)
            if not speech:
                return 0.0
            voiced = sum(float(seg["end"]) - float(seg["start"]) for seg in speech)
            total = float(len(wav)) / 16000.0
            return voiced / total if total > 0 else 0.0

        return silero_ratio, "silero"
    except Exception:
        import librosa

        def librosa_ratio(path: Path) -> float:
            y, sr = librosa.load(str(path), sr=16000)
            top_db = max(5, 30 - 5 * int(VAD_AGGRESSIVENESS))
            intervals = librosa.effects.split(y, top_db=top_db)
            voiced = sum(e - s for s, e in intervals)
            return float(voiced) / float(len(y)) if len(y) else 0.0

        return librosa_ratio, "librosa"


def run_vad_stage(
    records: list[dict[str, Any]],
    out_dir: Path,
    min_vad_ratio: float,
    move: bool,
    stats: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    vad_ratio_fn, backend = build_vad_ratio_fn()
    if stats is None:
        stats = Counter()
    pcm_cache_root = out_dir.parent / "pcm16"
    kept: list[dict[str, Any]] = []
    for record in records:
        stats["vad_input"] += 1
        try:
            src = Path(record.get("path_for_vad") or record["current_path"])
            pcm_src = stage_pcm16_path(pcm_cache_root, src, "vad")
            if not pcm_src.exists():
                ok, err = ensure_pcm16_wav(src, pcm_src)
                if not ok:
                    raise RuntimeError(f"failed to convert to pcm16 wav: {err or src}")
            src = pcm_src
            record["path_for_vad"] = str(src)
            y, sr = read_audio_numpy(src)
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)
            duration_sec = float(len(y)) / float(sr or 16000)
            if duration_sec < MIN_SEGMENT_DURATION:
                stats["vad_reject_too_short"] += 1
                record["status"] = "dropped_vad"
                record["active"] = False
                record["stage_status"]["vad"] = "dropped_too_short"
                record["note"] = f"vad_duration={duration_sec:.3f}"
                continue
            ratio = float(vad_ratio_fn(src))
            record["metrics"]["vad_ratio"] = ratio
            record["metrics"]["vad_backend"] = backend
            speech_duration = duration_sec * ratio
            silent_ratio_est = max(0.0, 1.0 - ratio)
            if silent_ratio_est > MAX_SILENCE_RATIO:
                stats["vad_reject_too_much_silence"] += 1
                record["status"] = "dropped_vad"
                record["active"] = False
                record["stage_status"]["vad"] = "dropped_too_much_silence"
                record["note"] = f"silent_ratio={silent_ratio_est:.3f}"
                continue
            if ratio < min_vad_ratio or speech_duration < MIN_SPEECH_DURATION:
                stats["vad_reject_no_speech"] += 1
                record["status"] = "dropped_vad"
                record["active"] = False
                record["stage_status"]["vad"] = "dropped_no_speech"
                record["note"] = f"vad_ratio={ratio:.3f}, speech_duration={speech_duration:.3f}"
                continue
            dst = out_dir / src.name
            copy_or_move(src, dst, move=move)
            record["current_path"] = str(dst)
            record["stage_paths"]["vad"] = str(dst)
            record["stage_status"]["vad"] = "passed"
            record["analysis_status"] = record.get("analysis_status") or "analysis_pass"
            stats["vad_pass"] += 1
            kept.append(record)
        except Exception as exc:
            stats["vad_reject_read_error"] += 1
            record["status"] = "dropped_vad_error"
            record["active"] = False
            record["stage_status"]["vad"] = f"error:{exc}"
            record["note"] = str(exc)
    return kept


def load_pyannote_pipeline(device: str = "auto"):
    try:
        from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore
    except Exception:
        return None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    last_exc: Exception | None = None
    for pipeline_id, revision in PYANNOTE_PIPELINE_CANDIDATES:
        try:
            kwargs: dict[str, Any] = {"revision": revision}
            if token:
                kwargs["token"] = token
            pipeline = PyannotePipeline.from_pretrained(pipeline_id, **kwargs)
            try:
                import torch

                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                pipeline.to(torch.device(device))
            except Exception:
                pass
            LOG.info("pyannote pipeline loaded: %s@%s", pipeline_id, revision)
            return pipeline
        except Exception as exc:
            last_exc = exc
            LOG.warning("pyannote pipeline %s could not be loaded: %s", pipeline_id, exc)
    if last_exc is not None:
        LOG.warning("pyannote pipeline could not be loaded: %s", last_exc)
    return None


def extract_diarization_annotation(result: Any):
    for attr in ("speaker_diarization", "exclusive_speaker_diarization", "annotation"):
        annotation = getattr(result, attr, None)
        if annotation is not None and hasattr(annotation, "itertracks"):
            return annotation
    if hasattr(result, "itertracks"):
        return result
    return None


def run_pyannote_stage(
    records: list[dict[str, Any]],
    out_dir: Path,
    enabled: bool,
    filter_multi_speaker: bool = True,
    batch_size: int = 1,
    device: str = "auto",
) -> list[dict[str, Any]]:
    ensure_dir(out_dir)
    if not enabled:
        for record in records:
            record["stage_status"]["pyannote"] = "skipped"
            dst = out_dir / Path(record["current_path"]).name
            copy_or_move(Path(record["current_path"]), dst, move=False)
            record["current_path"] = str(dst)
            record["stage_paths"]["diarized"] = str(dst)
        return records

    kept: list[dict[str, Any]] = []
    try:
        for batch in iter_batches(records, batch_size):
            pipeline = load_pyannote_pipeline(device=device)
            if pipeline is None:
                for record in batch:
                    record["stage_status"]["pyannote"] = "unavailable"
                    dst = out_dir / Path(record["current_path"]).name
                    copy_or_move(Path(record["current_path"]), dst, move=False)
                    record["current_path"] = str(dst)
                    record["stage_paths"]["diarized"] = str(dst)
                    kept.append(record)
                cleanup_runtime_cache()
                continue
            for record in batch:
                src = Path(record["current_path"])
                try:
                    y, sr = read_audio_numpy(src)
                    import numpy as np
                    import torch

                    arr = np.asarray(y, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    else:
                        arr = arr.T
                    waveform = torch.from_numpy(arr)
                    diarization_result = pipeline({"waveform": waveform, "sample_rate": int(sr)})
                    annotation = extract_diarization_annotation(diarization_result)
                    turns: list[dict[str, Any]] = []
                    speakers: set[str] = set()
                    if annotation is not None:
                        for turn, _, speaker in annotation.itertracks(yield_label=True):
                            turns.append(
                                {
                                    "start": float(turn.start),
                                    "end": float(turn.end),
                                    "speaker": str(speaker),
                                }
                            )
                            speakers.add(str(speaker))
                    else:
                        LOG.warning("pyannote result for %s did not expose diarization tracks", src)

                    diar_json = out_dir / f"{Path(record['current_path']).stem}.json"
                    diar_wav = out_dir / Path(record["current_path"]).name
                    ensure_dir(out_dir)
                    copy_or_move(src, diar_wav, move=False)
                    diar_json.write_text(
                        json.dumps({"sample_id": record["sample_id"], "turns": turns}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    record["stage_paths"]["diarized"] = str(diar_wav)
                    record["metrics"]["speaker_count"] = len(speakers)
                    record["metrics"]["speaker_turns"] = len(turns)
                    record["stage_status"]["pyannote"] = "passed"
                    record["current_path"] = str(diar_wav)
                    if filter_multi_speaker and len(speakers) > 1:
                        record["status"] = "dropped_multi_speaker"
                        record["active"] = False
                        record["note"] = f"speaker_count={len(speakers)}"
                        continue
                    kept.append(record)
                except Exception as exc:
                    record["stage_status"]["pyannote"] = f"error:{exc}"
                    record["status"] = "pyannote_error"
                    record["active"] = False
                    record["note"] = str(exc)
                finally:
                    # pyannote can keep native buffers alive across samples on low-VRAM GPUs.
                    # Reclaim aggressively after each record so later samples do not inherit pressure.
                    cleanup_runtime_cache()
    finally:
        try:
            del pipeline
        except Exception:
            pass
        cleanup_runtime_cache()
    return kept


def run_encoder_cli_batch(ref: Path, targets: list[Path], model: str | None = None, timeout: int = 240) -> dict[str, Any]:
    if not targets:
        return {"status": "ok", "sims": []}
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "demo" / "encoder_cli.py")]
    if model:
        cmd += ["--model", str(model)]
    cmd += ["--device", "auto"]
    cmd += ["--ref", Path(ref).resolve().as_posix(), "--targets"] + [Path(t).resolve().as_posix() for t in targets]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        out_text = (proc.stdout or "").strip()
        if not out_text:
            LOG.warning("encoder_cli returned empty stdout")
            return {}
        data = json.loads(out_text)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as exc:
        LOG.warning("encoder_cli call failed: %s", exc)
        return {}


def resolve_voiceprint_reference_paths(speaker_ref: Path) -> list[Path]:
    """Resolve one reference file or a directory of reference files."""
    if speaker_ref.is_dir():
        candidates = sorted(
            p for p in speaker_ref.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"}
        )
        return candidates
    if speaker_ref.exists():
        return [speaker_ref]
    return []


def resolve_voiceprint_reference_groups(speaker_ref: Path) -> tuple[list[Path], list[Path]]:
    """Return positive and negative reference paths.

    Layout supported:
      refs/
        positive/
        negative_wrong_speaker/
        negative_music/
        negative_nonspeech/
    """
    if speaker_ref.is_file():
        return [speaker_ref], []
    if not speaker_ref.exists():
        return [], []

    positive_dir = speaker_ref / "positive"
    positive_refs: list[Path] = []
    negative_refs: list[Path] = []

    if positive_dir.is_dir():
        positive_refs = sorted(
            p for p in positive_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"}
        )
    else:
        for p in speaker_ref.iterdir():
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"}:
                positive_refs.append(p)

    for child in speaker_ref.iterdir():
        if not child.is_dir():
            continue
        if not child.name.lower().startswith("negative"):
            continue
        negative_refs.extend(
            p for p in child.rglob("*")
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"}
        )

    return positive_refs, negative_refs


def write_voiceprint_input(vad_pass_items: list[dict[str, Any]], manifest_path: Path, pcm_cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    ensure_dir(manifest_path.parent)
    ensure_dir(pcm_cache_root)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(vad_pass_items, 1):
        if not item.get("segment_id"):
            item["segment_id"] = f"seg_{i:05d}"
        src = Path(item.get("path_for_verify") or item.get("path_for_vad") or item.get("current_path") or item.get("path") or "")
        item["path_for_verify"] = str(src.resolve().as_posix())
        pcm_path = stage_pcm16_path(pcm_cache_root, src, "voiceprint")
        if not pcm_path.exists():
            ok, err = ensure_pcm16_wav(src, pcm_path)
            if ok:
                item["path_for_verify"] = str(pcm_path.resolve().as_posix())
            else:
                item["voiceprint_error"] = err or "pcm16_conversion_failed"
        else:
            item["path_for_verify"] = str(pcm_path.resolve().as_posix())
        rows.append({
            "segment_id": item["segment_id"],
            "path": item["path_for_verify"],
        })
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def write_reference_manifest(ref_items: list[Path], manifest_path: Path) -> list[dict[str, Any]]:
    ensure_dir(manifest_path.parent)
    rows: list[dict[str, Any]] = []
    for idx, ref in enumerate(ref_items, 1):
        ref = Path(ref)
        if not ref.exists():
            continue
        rows.append({
            "segment_id": f"ref_{idx:04d}_{safe_name(ref.stem, 32)}",
            "path": str(ref.resolve().as_posix()),
        })
    write_jsonl(rows, manifest_path)
    return rows


def run_encoder_cli_manifest(
    ref: Path | None,
    manifest: Path,
    model: str | None = None,
    timeout: int = 240,
    ref_manifest: Path | None = None,
) -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "demo" / "encoder_cli.py")]
    if model:
        cmd += ["--model", str(model)]
    cmd += ["--device", "auto"]
    if ref_manifest is not None:
        cmd += ["--ref-manifest", str(Path(ref_manifest).resolve())]
    elif ref is not None:
        cmd += ["--ref", Path(ref).resolve().as_posix()]
    else:
        return {"status": "error", "error": "missing ref or ref_manifest", "results": []}
    cmd += ["--manifest", str(Path(manifest).resolve())]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        out_text = (proc.stdout or "").strip()
        if not out_text:
            return {"status": "error", "error": "empty stdout", "results": []}
        data = json.loads(out_text)
        if not isinstance(data, dict):
            return {"status": "error", "error": "non-dict output", "results": []}
        data.setdefault("results", data.get("sims", []))
        return data
    except Exception as exc:
        return {"status": "error", "error": str(exc), "results": []}


def collect_best_voiceprint_results(
    ref_items: list[Path],
    manifest_path: Path,
    model_path: str | None,
) -> tuple[dict[str, dict[str, Any]], list[str], bool]:
    best_results: dict[str, dict[str, Any]] = {}
    cli_errors: list[str] = []
    any_ok = False
    for ref_item in ref_items:
        payload = run_encoder_cli_manifest(ref_item, manifest_path, model=model_path)
        payload_status = str(payload.get("status") or "error")
        if payload_status == "ok":
            any_ok = True
        else:
            cli_errors.append(payload_status)
        results = payload.get("results") or payload.get("sims") or []
        for row in results:
            if not isinstance(row, dict):
                continue
            segment_id = str(row.get("segment_id") or "").strip()
            if not segment_id:
                continue
            current = best_results.get(segment_id)
            ok = bool(row.get("ok", row.get("status") in {"ok", "success"}))
            if current is None:
                best_results[segment_id] = dict(row)
                best_results[segment_id]["ref_name"] = ref_item.name
                best_results[segment_id]["ref_key"] = path_key(ref_item)
                continue
            current_ok = bool(current.get("ok", current.get("status") in {"ok", "success"}))
            current_sim = current.get("sim")
            row_sim = row.get("sim")
            try:
                current_sim_f = float(current_sim) if current_sim is not None else float("-inf")
            except Exception:
                current_sim_f = float("-inf")
            try:
                row_sim_f = float(row_sim) if row_sim is not None else float("-inf")
            except Exception:
                row_sim_f = float("-inf")
            if ok and (not current_ok or row_sim_f >= current_sim_f):
                best_results[segment_id] = dict(row)
                best_results[segment_id]["ref_name"] = ref_item.name
                best_results[segment_id]["ref_key"] = path_key(ref_item)
    return best_results, cli_errors, any_ok


def apply_voiceprint_results(vad_pass_items: list[dict[str, Any]], voiceprint_json: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    by_id = {str(item["segment_id"]): item for item in vad_pass_items if item.get("segment_id")}
    stats: Counter[str] = Counter()
    stats["voiceprint_input"] = len(vad_pass_items)
    results = voiceprint_json.get("results") or voiceprint_json.get("sims") or []
    stats["voiceprint_result_count"] = len(results)
    seen_ids: set[str] = set()

    for r in results:
        if not isinstance(r, dict):
            continue
        segment_id = str(r.get("segment_id") or "").strip()
        if not segment_id:
            continue
        seen_ids.add(segment_id)
        item = by_id.get(segment_id)
        if item is None:
            continue
        ok = bool(r.get("ok", r.get("status") in {"ok", "success"}))
        if not ok:
            item["speaker_similarity"] = None
            item["voiceprint_status"] = "voiceprint_compute_error"
            item["voiceprint_error"] = r.get("error")
            item["stage_status"]["voiceprint"] = "dropped"
            item["stage_status"]["voiceprint_error"] = str(r.get("error") or "voiceprint_compute_error")
            stats["voiceprint_compute_error"] += 1
            stats["verified_reject"] += 1
            continue
        sim = r.get("sim")
        if sim is None or sim == "":
            item["speaker_similarity"] = None
            item["voiceprint_status"] = "voiceprint_missing_similarity"
            item["stage_status"]["voiceprint"] = "missing_similarity"
            stats["voiceprint_missing_similarity"] += 1
            stats["verified_reject"] += 1
            continue
        try:
            sim_f = float(sim)
        except Exception:
            item["speaker_similarity"] = None
            item["voiceprint_status"] = "voiceprint_missing_similarity"
            item["stage_status"]["voiceprint"] = "missing_similarity"
            stats["voiceprint_missing_similarity"] += 1
            stats["verified_reject"] += 1
            continue
        item["speaker_similarity"] = sim_f
        item["voiceprint_status"] = "voiceprint_scored"
        item["stage_status"]["voiceprint"] = "scored"
        stats["voiceprint_written"] += 1
        if sim_f >= VOICEPRINT_PASS_TH:
            item["voiceprint_status"] = "voiceprint_pass"
            item["stage_status"]["voiceprint"] = "passed"
            stats["verified_pass"] += 1
        elif sim_f >= VOICEPRINT_UNCERTAIN_TH:
            item["voiceprint_status"] = "voiceprint_uncertain"
            item["stage_status"]["voiceprint"] = "review"
            item["final_score_penalty"] = float(item.get("final_score_penalty", 0.0) or 0.0) + 10.0
            stats["verified_uncertain"] += 1
        else:
            item["voiceprint_status"] = "voiceprint_reject"
            item["stage_status"]["voiceprint"] = "dropped"
            item["active"] = False
            item["status"] = "dropped_voiceprint"
            stats["verified_reject"] += 1

    for segment_id, item in by_id.items():
        if segment_id not in seen_ids:
            item["speaker_similarity"] = None
            item["voiceprint_status"] = "voiceprint_missing_result"
            item["stage_status"]["voiceprint"] = "missing_result"
            stats["voiceprint_missing_result"] += 1
            stats["verified_reject"] += 1

    return vad_pass_items, stats


def run_voiceprint_stage(
    records: list[dict[str, Any]],
    out_dir: Path,
    speaker_ref: str | None,
    enabled: bool,
    speaker_thres: float,
    batch_size: int,
    model_path: str | None,
    move: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    ensure_dir(out_dir)
    pcm_cache_root = out_dir.parent / "pcm16"
    stats: Counter[str] = Counter()
    if not enabled or not speaker_ref:
        for record in records:
            record["stage_status"]["voiceprint"] = "skipped"
            record["speaker_similarity"] = None
            record["voiceprint_status"] = "voiceprint_skipped"
            dst = out_dir / Path(record["current_path"]).name
            copy_or_move(Path(record["current_path"]), dst, move=False)
            record["current_path"] = str(dst)
            record["stage_paths"]["verified"] = str(dst)
        stats["voiceprint_cli_status"] = "skipped"
        stats["voiceprint_input"] = len(records)
        stats["voiceprint_result_count"] = 0
        return records, stats

    ref_path = Path(speaker_ref)
    refs_dir = out_dir.parent / "refs"
    ensure_dir(refs_dir)
    positive_refs, negative_refs = resolve_voiceprint_reference_groups(ref_path)
    if not positive_refs:
        LOG.warning("no voiceprint reference files found at %s", ref_path)
        for record in records:
            record["stage_status"]["voiceprint"] = "missing_reference"
            record["speaker_similarity"] = None
            record["voiceprint_status"] = "voiceprint_missing_reference"
            dst = out_dir / Path(record["current_path"]).name
            copy_or_move(Path(record["current_path"]), dst, move=False)
            record["current_path"] = str(dst)
            record["stage_paths"]["verified"] = str(dst)
        stats["voiceprint_cli_status"] = "missing_reference"
        stats["voiceprint_input"] = len(records)
        stats["voiceprint_result_count"] = 0
        return records, stats

    ref_to_use_list: list[Path] = []
    for src in positive_refs:
        try:
            conv_ref = stage_pcm16_path(refs_dir, src, "ref")
            if not conv_ref.exists():
                ok, err = ensure_pcm16_wav(src, conv_ref)
                if not ok:
                    raise RuntimeError(err or f"failed to convert reference {src}")
            ref_to_use_list.append(conv_ref)
        except Exception as exc:
            LOG.warning("reference conversion failed for %s, using original file: %s", src, exc)
            ref_to_use_list.append(src)

    negative_ref_to_use_list: list[Path] = []
    for src in negative_refs:
        try:
            conv_ref = stage_pcm16_path(refs_dir, src, "neg")
            if not conv_ref.exists():
                ok, err = ensure_pcm16_wav(src, conv_ref)
                if not ok:
                    raise RuntimeError(err or f"failed to convert negative reference {src}")
            negative_ref_to_use_list.append(conv_ref)
        except Exception as exc:
            LOG.warning("negative reference conversion failed for %s, using original file: %s", src, exc)
            negative_ref_to_use_list.append(src)

    manifest_path = out_dir.parent / "debug" / "voiceprint_input.jsonl"
    rows = write_voiceprint_input(records, manifest_path, pcm_cache_root)
    stats["voiceprint_input"] = len(rows)
    review_uncertain_dir = out_dir.parent / "review_uncertain"
    ensure_dir(review_uncertain_dir)

    positive_manifest = out_dir.parent / "debug" / "voiceprint_positive_refs.jsonl"
    write_reference_manifest(ref_to_use_list, positive_manifest)
    positive_payload = run_encoder_cli_manifest(None, manifest_path, model=model_path, ref_manifest=positive_manifest)
    positive_status = str(positive_payload.get("status") or "error")
    positive_results = positive_payload.get("results") or positive_payload.get("sims") or []
    if not isinstance(positive_results, list):
        positive_results = []
    cli_status = positive_status
    merged_payload = {"status": cli_status, "results": [r for r in positive_results if isinstance(r, dict)]}
    updated_records, apply_stats = apply_voiceprint_results(records, merged_payload)
    stats.update(apply_stats)
    stats["voiceprint_input"] = len(rows)
    stats["voiceprint_cli_status"] = cli_status
    stats["voiceprint_result_count"] = len(merged_payload.get("results", []))
    negative_best: dict[str, float] = {}
    negative_statuses: list[str] = []
    for neg_ref in negative_ref_to_use_list:
        neg_payload = run_encoder_cli_manifest(neg_ref, manifest_path, model=model_path)
        neg_status = str(neg_payload.get("status") or "error")
        negative_statuses.append(neg_status)
        neg_results = neg_payload.get("results") or neg_payload.get("sims") or []
        if not isinstance(neg_results, list):
            continue
        for row in neg_results:
            if not isinstance(row, dict):
                continue
            segment_id = str(row.get("segment_id") or "").strip()
            if not segment_id:
                continue
            if not bool(row.get("ok", row.get("status") in {"ok", "success"})):
                continue
            sim = row.get("sim")
            if sim is None or sim == "":
                continue
            try:
                sim_f = float(sim)
            except Exception:
                continue
            current = negative_best.get(segment_id)
            if current is None or sim_f > current:
                negative_best[segment_id] = sim_f
    stats["voiceprint_negative_cli_status"] = "ok" if negative_ref_to_use_list and any(s == "ok" for s in negative_statuses) else ("skipped" if not negative_ref_to_use_list else (negative_statuses[0] if negative_statuses else "error"))
    stats["voiceprint_negative_result_count"] = len(negative_best)

    kept: list[dict[str, Any]] = []
    for record in updated_records:
        voiceprint_status = str(record.get("voiceprint_status") or "")
        src = Path(record.get("path_for_verify") or record.get("path_for_vad") or record.get("current_path") or "")
        segment_id = str(record.get("segment_id") or "").strip()
        negative_sim = negative_best.get(segment_id)
        record["speaker_negative_similarity"] = negative_sim
        if record.get("speaker_similarity") is not None and negative_sim is not None:
            record["speaker_margin"] = float(record.get("speaker_similarity") or 0.0) - float(negative_sim)
        else:
            record["speaker_margin"] = None
        record["negative_reference_available"] = bool(negative_ref_to_use_list)
        if voiceprint_status == "voiceprint_pass":
            dst = out_dir / src.name
            copy_or_move(src, dst, move=move)
            record["current_path"] = str(dst)
            record["stage_paths"]["verified"] = str(dst)
            record["stage_status"]["voiceprint"] = "passed"
            record["active"] = True
            record["status"] = "voiceprint_pass"
            kept.append(record)
        elif voiceprint_status == "voiceprint_uncertain":
            dst = review_uncertain_dir / src.name
            copy_or_move(src, dst, move=False)
            record["current_path"] = str(dst)
            record["stage_paths"]["review_uncertain"] = str(dst)
            record["stage_paths"]["verified"] = str(dst)
            record["stage_status"]["voiceprint"] = "review"
            record["active"] = False
            record["status"] = "voiceprint_review"
        else:
            record["active"] = False
            record["status"] = "dropped_voiceprint"
            record["stage_paths"]["verified"] = str(src)
            if voiceprint_status in {"voiceprint_compute_error", "voiceprint_missing_similarity", "voiceprint_missing_result", "voiceprint_missing_reference"}:
                record["stage_status"]["voiceprint"] = "dropped"
            else:
                record["stage_status"]["voiceprint"] = "dropped"
    cleanup_runtime_cache()
    return kept, stats


def transcribe_with_whisper(records: list[dict[str, Any]], batch_size: int = 2) -> dict[str, str]:
    try:
        results: dict[str, str] = {}
        pending: list[dict[str, Any]] = []
        for record in records:
            key = str(record.get("sample_id") or Path(record["current_path"]).stem)
            if record.get("transcript") is not None:
                results[key] = str(record.get("transcript") or "")
            else:
                pending.append(record)
        if not pending:
            return results

        whisper_py = Path(__file__).resolve().parents[1] / "demo" / "whisper.py"
        module = load_module_from_path("mockingbird_whisper_probe", whisper_py)
        model_path = str(CFG.get("whisper_model_dir") or os.environ.get("MOCKINGBIRD_WHISPER_MODEL") or "large-v3")
        model = module.load_whisper_model(model_spec=model_path)
        for batch in iter_batches(pending, batch_size):
            for record in batch:
                try:
                    key = str(record.get("sample_id") or Path(record["current_path"]).stem)
                    stats = module.transcribe_with_stats(
                        Path(record["current_path"]),
                        model,
                        beam_size=int(CFG.get("whisper_beam", 1) or 1),
                        vad_filter=bool(CFG.get("whisper_vad", True)),
                        language=CFG.get("whisper_lang") or "zh",
                    )
                    text = str(stats.get("text", "") or "")
                    record["metrics"]["whisper_text_len"] = len(normalize_match_text(text))
                    record["metrics"]["whisper_audio_duration_sec"] = float(stats.get("audio_duration_sec", 0.0) or 0.0)
                    record["metrics"]["whisper_transcript_duration_sec"] = float(stats.get("transcript_duration_sec", 0.0) or 0.0)
                    record["metrics"]["whisper_avg_logprob"] = stats.get("avg_logprob")
                    record["metrics"]["whisper_no_speech_prob"] = stats.get("no_speech_prob")
                    record["metrics"]["whisper_quality"] = stats.get("quality", {})
                    results[key] = text
                except Exception as exc:
                    LOG.warning("transcription failed %s: %s", record["current_path"], exc)
                    key = str(record.get("sample_id") or Path(record["current_path"]).stem)
                    results[key] = ""
                finally:
                    cleanup_runtime_cache()
        return results
    except Exception as exc:
        LOG.warning("whisper unavailable, falling back to empty transcripts: %s", exc)
        return {str(record.get("sample_id") or Path(record["current_path"]).stem): "" for record in records}


def compute_transcript_confidence(text: str, stats: dict[str, Any] | None = None) -> float:
    if not text:
        return 0.0
    if is_low_value_transcript(text):
        return 0.0
    if not contains_chinese(text):
        return 0.0
    clean = normalize_match_text(text)
    if len(clean) < WHISPER_MIN_TEXT_LEN:
        return 0.0
    base = float(min(1.0, len(clean) / 18.0))
    if stats:
        quality = stats.get("whisper_quality")
        if isinstance(quality, dict):
            if not quality.get("accepted", True):
                return 0.0
            if quality.get("uncertain"):
                base *= 0.85
        no_speech_prob = stats.get("whisper_no_speech_prob")
        avg_logprob = stats.get("whisper_avg_logprob")
        transcript_duration = float(stats.get("whisper_transcript_duration_sec", 0.0) or 0.0)
        audio_duration = float(stats.get("whisper_audio_duration_sec", 0.0) or 0.0)
        if no_speech_prob is not None:
            base *= max(0.0, 1.0 - float(no_speech_prob))
        if avg_logprob is not None:
            base *= max(0.0, min(1.0, (float(avg_logprob) + 1.5) / 1.5))
        if audio_duration > 0.0:
            ratio = transcript_duration / max(audio_duration, 1e-6)
            if ratio < WHISPER_MIN_DURATION_RATIO or ratio > WHISPER_MAX_DURATION_RATIO:
                return 0.0
    return float(min(1.0, max(0.0, base)))


def normalize_snr(db: float) -> float:
    return min(1.0, max(0.0, (db + 10.0) / 40.0))


def normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def is_low_value_transcript(text: str) -> bool:
    blob = normalize_match_text(text)
    if not blob:
        return True
    if len(blob) < 4:
        return True
    return any(token in blob for token in TRANSCRIPT_REJECT_PATTERNS)


def is_interview_like_source(record: dict[str, Any]) -> bool:
    blob = normalize_match_text(record.get("source_name", "")) + normalize_match_text(record.get("sample_id", ""))
    has_hint = any(token in blob for token in INTERVIEW_TITLE_HINTS)
    has_risk = any(token in blob for token in VOICEPRINT_REVIEW_TITLE_HINTS)
    return bool(has_hint and not has_risk)


def compute_voiceprint_threshold(record: dict[str, Any], base_thres: float) -> float:
    # Keep the current speaker verification gate strict and stable.
    # Interview-like sources can still be reviewed downstream, but we do not
    # relax the hard threshold automatically.
    return max(float(base_thres), float(VOICEPRINT_PASS_TH))


def should_keep_voiceprint_review(record: dict[str, Any], sim: float, threshold: float) -> bool:
    if not is_interview_like_source(record):
        return False
    snr_db = float(record["metrics"].get("snr_db", -10.0))
    vad_ratio = float(record["metrics"].get("vad_ratio", 0.0))
    reverb = float(record["metrics"].get("reverb_score", 1.0))
    if sim < 0.58:
        return False
    return snr_db >= 15.0 and vad_ratio >= 0.95 and reverb <= 0.50


def quality_score(record: dict[str, Any]) -> float:
    snr_db = float(record["metrics"].get("snr_db", -10.0))
    vad_ratio = float(record["metrics"].get("vad_ratio", 0.0))
    clip_ratio = float(record["metrics"].get("clipping_ratio", 0.0))
    reverb = float(record["metrics"].get("reverb_score", 0.0))
    similarity = record.get("speaker_similarity")
    trans_conf = float(record.get("transcript_conf", 0.0))
    speaker_component = float(similarity) if similarity is not None else 0.60
    snr_component = normalize_snr(snr_db)
    vad_component = max(0.0, min(1.0, vad_ratio))
    transcript_component = max(0.0, min(1.0, trans_conf))
    clip_penalty = min(1.0, clip_ratio * 20.0)
    pre_b_status = str(record.get("stage_status", {}).get("pre_b") or "")
    demucs_status = str(record.get("stage_status", {}).get("demucs_quality") or record.get("stage_status", {}).get("demucs") or "")
    route = str(record.get("stage_status", {}).get("pre_b") or "")
    voiceprint_status = str(record.get("voiceprint_status") or record.get("stage_status", {}).get("voiceprint") or "")
    analysis_status = str(record.get("analysis_status") or record.get("stage_status", {}).get("quality_analyzer") or "")
    music_like = bool(record.get("metrics", {}).get("music_like"))

    score = (
        QUALITY_WEIGHTS["speaker"] * speaker_component
        + QUALITY_WEIGHTS["snr"] * snr_component
        + QUALITY_WEIGHTS["vad"] * vad_component
        + QUALITY_WEIGHTS["transcript"] * transcript_component
    )
    score -= QUALITY_PENALTIES["reverb"] * min(1.0, reverb)
    score -= QUALITY_PENALTIES["clip"] * clip_penalty
    if pre_b_status in {"uncertain", "fallback_original"}:
        score -= 0.08
    if demucs_status in {"fallback_original", "failed"}:
        score -= 0.06
    if route and route not in {"pass_vocals", "fallback_original"}:
        score -= 0.04
    if music_like:
        score -= 0.15
    if voiceprint_status in {"voiceprint_pass", "passed"}:
        score += 0.12
    elif voiceprint_status in {"voiceprint_uncertain", "review"}:
        score += 0.04
    elif voiceprint_status in {"voiceprint_skipped", "skipped"}:
        score -= 0.05
    if analysis_status == "analysis_uncertain":
        score -= 0.08
    elif analysis_status == "analysis_error_fallback":
        score -= 0.12
    elif analysis_status == "analysis_pass":
        score += 0.02
    return float(max(0.0, min(1.0, score)))


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_speaker_similarity(record: dict[str, Any]) -> float:
    value = record.get("speaker_similarity")
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0
    return 0.0


def get_speaker_similarity(record: dict[str, Any]) -> float | None:
    value = record.get("speaker_similarity")
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def export_verified_pass_debug(
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    selected_ids = {str(item.get("segment_id") or item.get("sample_id") or "") for item in selected}
    rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records, 1):
        segment_id = str(record.get("segment_id") or record.get("sample_id") or f"verified_{idx:05d}")
        src = Path(record.get("current_path") or record.get("path_for_verify") or record.get("path_for_vad") or "")
        if not src.exists():
            continue
        dst = out_dir / f"{idx:04d}_{src.name}"
        copy_or_move(src, dst, move=False)

        transcript = str(record.get("transcript", "") or "")
        metrics = record.get("metrics", {}) or {}
        final_reasons = list(record.get("final_reasons") or [])
        if segment_id in selected_ids:
            final_reject_reason = "final_selected"
        elif final_reasons:
            final_reject_reason = ";".join(dict.fromkeys(str(r) for r in final_reasons if r))
        else:
            final_reject_reason = str(record.get("status") or "final_rejected")

        row = {
            "segment_id": segment_id,
            "path": str(dst.resolve().as_posix()),
            "source_path": str(src.resolve().as_posix()),
            "speaker_similarity": record.get("speaker_similarity"),
            "voiceprint_status": record.get("voiceprint_status"),
            "final_reject_reason": final_reject_reason,
            "music_score": compute_music_score(record),
            "music_like": bool(metrics.get("music_like")),
            "singing_like": bool(metrics.get("singing_like") or record.get("singing_like")),
            "asr_text": transcript,
            "asr_no_speech_prob": metrics.get("whisper_no_speech_prob"),
            "asr_text_len": len(normalize_match_text(transcript)),
            "silent_ratio": metrics.get("silent_ratio"),
            "duration": float(record.get("duration_sec", 0.0) or 0.0),
            "vad_ratio": metrics.get("vad_ratio"),
            "analysis_status": record.get("analysis_status"),
            "analysis_score": record.get("analysis_score"),
            "final_score": record.get("final_score"),
            "sample_id": record.get("sample_id"),
        }
        rows.append(row)

    jsonl_path = out_dir / "verified_pass.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = out_dir / "verified_pass.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")


def route_final_speaker(speaker_sim: float) -> str:
    if speaker_sim >= HIGH_CONF_SPEAKER_SIM:
        return "high_confidence"
    if speaker_sim >= TARGET_SIM_TH:
        return "final_candidate"
    if speaker_sim >= SPEAKER_SIM_UNCERTAIN:
        return "uncertain"
    return "reject"


def inspect_final_audio(record: dict[str, Any]) -> tuple[float, float, float, float]:
    path = Path(record.get("current_path") or record.get("path") or "")
    if not path.exists():
        raise FileNotFoundError(str(path))
    y, sr = read_audio_numpy(path)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    duration = float(len(y)) / float(sr or 16000)
    return duration, rms_db(y), peak_dbfs(y), silent_ratio(y, sr)


def clean_speech_gate(record: dict[str, Any]) -> tuple[str, str]:
    sim = get_speaker_similarity(record)
    if sim is None:
        return "reject", "missing_similarity"
    voiceprint_status = str(record.get("voiceprint_status") or record.get("stage_status", {}).get("voiceprint") or "")
    if voiceprint_status not in {"voiceprint_pass", "passed"}:
        return "reject", "voiceprint_not_pass"

    metrics = record.get("metrics", {}) or {}
    music_score = float(record.get("music_score", compute_music_score(record)) or compute_music_score(record))
    silent_ratio_value = float(record.get("silent_ratio", metrics.get("silent_ratio", 1.0)) or metrics.get("silent_ratio", 1.0) or 0.0)
    vad_ratio = float(record.get("vad_ratio", metrics.get("vad_ratio", 0.0)) or metrics.get("vad_ratio", 0.0) or 0.0)
    asr_len = int(record.get("asr_text_len", metrics.get("whisper_text_len", 0)) or metrics.get("whisper_text_len", 0) or 0)
    no_speech_prob = metrics.get("whisper_no_speech_prob")
    singing_like = bool(record.get("singing_like") or metrics.get("singing_like"))
    num_speakers = int(metrics.get("speaker_count", record.get("speaker_count", 1)) or 1)
    margin = record.get("speaker_margin")
    negative_available = bool(record.get("negative_reference_available"))
    target_source = str(record.get("source_name") or record.get("sample_id") or "")
    interview_like = is_interview_like_source(record)

    if music_score > MAX_MUSIC_SCORE:
        return "reject", "music_score_high"
    if singing_like:
        return "reject", "singing_like"
    if num_speakers > 1:
        return "reject", "multi_speaker"
    if silent_ratio_value > MAX_SILENT_RATIO:
        return "reject", "too_much_silence"
    if vad_ratio < MIN_VAD_RATIO:
        return "reject", "vad_ratio_low"
    if asr_len < MIN_ASR_TEXT_LEN:
        if no_speech_prob is not None and float(no_speech_prob) >= 0.60:
            return "reject", "asr_empty_high_no_speech"
        if sim >= TARGET_SIM_TH and vad_ratio >= MIN_VAD_RATIO and music_score <= MAX_MUSIC_SCORE and silent_ratio_value <= MAX_SILENT_RATIO:
            return "manual_transcribe", "asr_empty_but_voiceprint_high"
        return "reject", "asr_empty"
    if negative_available:
        if margin is None:
            return "reject", "negative_margin_missing"
        if float(margin) < NEGATIVE_MARGIN_TH:
            return "reject", "negative_margin_low"
    if sim < TARGET_SIM_TH:
        return "reject", "speaker_similarity_low"
    if interview_like and not target_source:
        return "manual_transcribe", "interview_like_review"
    return "final", "ok"


def select_final_records(
    records: list[dict[str, Any]],
    max_files: int = FINAL_MAX_FILES,
    target_total_seconds: float = FINAL_TARGET_TOTAL_SECONDS,
    min_speaker_sim: float = MIN_FINAL_SPEAKER_SIM,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    final_reject_reasons: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []

    for record in records:
        if not record.get("active", True):
            continue
        record["final_reasons"] = []
        try:
            duration_sec, measured_rms_db, measured_peak_db, measured_silent_ratio = inspect_final_audio(record)
        except FileNotFoundError:
            final_reject_reasons["file_not_found"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("file_not_found")
            record["note"] = f"missing_path:{record.get('current_path') or record.get('path')}"
            continue
        except Exception as exc:
            final_reject_reasons["cannot_read_audio"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("cannot_read_audio")
            record["note"] = f"cannot_read_audio:{exc}"
            continue

        raw_speaker_sim = record.get("speaker_similarity")
        if raw_speaker_sim is None or raw_speaker_sim == "":
            final_reject_reasons["missing_similarity"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("missing_similarity")
            record["note"] = "speaker_similarity missing"
            continue
        try:
            speaker_sim = float(raw_speaker_sim)
        except Exception:
            final_reject_reasons["missing_similarity"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("missing_similarity")
            record["note"] = f"speaker_similarity invalid:{raw_speaker_sim}"
            continue
        rms_db = float(record.get("metrics", {}).get("rms_db", measured_rms_db))
        peak_db = float(record.get("metrics", {}).get("peak_dbfs", measured_peak_db))
        silent_ratio_value = float(record.get("metrics", {}).get("silent_ratio", measured_silent_ratio))
        duration_sec = float(record.get("duration_sec", 0.0) or duration_sec)
        final_route = route_final_speaker(speaker_sim)
        record["final_route"] = final_route

        if duration_sec < FINAL_MIN_DURATION_SEC:
            final_reject_reasons["too_short"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("too_short")
            record["note"] = f"duration={duration_sec:.2f}s below minimum {FINAL_MIN_DURATION_SEC:.1f}s"
            continue
        if duration_sec > FINAL_MAX_DURATION_SEC:
            final_reject_reasons["too_long"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("too_long")
            record["note"] = f"duration={duration_sec:.2f}s above maximum {FINAL_MAX_DURATION_SEC:.1f}s"
            continue
        if rms_db < FINAL_MIN_RMS_DB:
            final_reject_reasons["too_quiet"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("too_quiet")
            record["note"] = f"rms_db={rms_db:.2f} below minimum {FINAL_MIN_RMS_DB:.1f}dB"
            continue
        if silent_ratio_value > FINAL_MAX_SILENT_RATIO:
            final_reject_reasons["too_much_silence"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("too_much_silence")
            record["note"] = f"silent_ratio={silent_ratio_value:.3f} above maximum {FINAL_MAX_SILENT_RATIO:.2f}"
            continue
        if final_route == "reject":
            final_reject_reasons["speaker_sim_too_low"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_reasons"].append("speaker_sim_too_low")
            record["note"] = f"speaker_sim={speaker_sim:.4f} below reject threshold {SPEAKER_SIM_REJECT:.2f}"
            continue

        transcript = str(record.get("transcript", "") or "")
        whisper_quality = record.get("metrics", {}).get("whisper_quality")
        metrics = record.get("metrics", {})
        pre_b_status = str(record.get("stage_status", {}).get("pre_b") or "")
        demucs_quality = str(record.get("stage_status", {}).get("demucs_quality") or record.get("stage_status", {}).get("demucs") or "")
        route = str(record.get("stage_status", {}).get("vocal_separator") or record.get("stage_status", {}).get("demucs_quality") or "")
        voiceprint_status = str(record.get("voiceprint_status") or record.get("stage_status", {}).get("voiceprint") or "")
        analysis_status = str(record.get("analysis_status") or record.get("stage_status", {}).get("quality_analyzer") or "")
        music_like = bool(metrics.get("music_like"))
        singing_like = bool(metrics.get("singing_like") or record.get("singing_like"))
        speaker_count = int(metrics.get("speaker_count", record.get("speaker_count", 1)) or 1)
        overlap_ratio = float(metrics.get("overlap_ratio", record.get("overlap_ratio", 0.0)) or 0.0)
        target_speaker_ratio = float(metrics.get("target_speaker_ratio", record.get("target_speaker_ratio", 1.0)) or 1.0)
        whisper_text_len = len(normalize_match_text(transcript))
        whisper_no_speech_prob = metrics.get("whisper_no_speech_prob")
        final_score = 0.0
        final_score += speaker_sim * 60.0
        if 2.0 <= duration_sec <= 10.0:
            final_score += 15.0
        if -32.0 <= rms_db <= -12.0:
            final_score += 10.0
        if silent_ratio_value < 0.45:
            final_score += 10.0
        if voiceprint_status in {"voiceprint_pass", "passed"}:
            final_score += 8.0
        elif voiceprint_status in {"voiceprint_uncertain", "voiceprint_review", "review"}:
            final_score -= 8.0
            record["final_reasons"].append("voiceprint_uncertain")
        if pre_b_status == "pass_vocals":
            final_score += 4.0
        if demucs_quality == "passed":
            final_score += 3.0
        if not is_low_value_transcript(transcript):
            final_score += 3.0
        if speaker_sim >= HIGH_CONF_SPEAKER_SIM:
            final_score += 5.0
        elif speaker_sim >= SPEAKER_SIM_PASS:
            final_score += 2.0
        if analysis_status == "analysis_uncertain":
            final_score -= 8.0
            record["final_reasons"].append("analysis_uncertain")
        elif analysis_status == "analysis_error_fallback":
            final_score -= 12.0
            record["final_reasons"].append("analysis_error_fallback")
        elif analysis_status == "analysis_pass":
            final_score += 0.0
        if route_final_speaker(speaker_sim) == "uncertain":
            final_score -= 3.0
        if is_low_value_transcript(transcript):
            final_score -= 5.0
            record["final_reasons"].append("low_value_transcript")
        if pre_b_status == "pass_vocals":
            final_score += 0.0
        elif pre_b_status == "uncertain":
            final_score -= 8.0
            record["final_reasons"].append("pre_b_uncertain")
        elif pre_b_status == "fallback_original":
            final_score -= 5.0
            record["final_reasons"].append("demucs_fallback")
        elif pre_b_status:
            final_score -= 3.0
            record["final_reasons"].append(f"pre_b_{pre_b_status}")
        if demucs_quality == "passed":
            final_score += 0.0
        elif demucs_quality in {"fallback_original", "failed"}:
            final_score -= 5.0
            record["final_reasons"].append("demucs_failed_fallback")
        if route in {"fallback", "uncertain"}:
            final_score -= 2.0
        if music_like:
            record["final_reasons"].append("music_like")
        if singing_like:
            record["final_reasons"].append("singing_like")
        if speaker_count > 1:
            record["final_reasons"].append("multi_speaker")
        if overlap_ratio > 0.05:
            record["final_reasons"].append("overlap_speech")
        if target_speaker_ratio < 0.85:
            record["final_reasons"].append("target_speaker_ratio_low")
        if whisper_text_len < 2:
            record["final_reasons"].append("asr_empty")
        if whisper_no_speech_prob is not None and float(whisper_no_speech_prob) > 0.45:
            record["final_reasons"].append("asr_no_speech_high")
        if isinstance(whisper_quality, dict) and not whisper_quality.get("accepted", True):
            record["final_reasons"].append("whisper_quality_rejected")
        if final_route == "high_confidence":
            final_score += 0.0
        elif final_route == "final_candidate":
            final_score += 0.0
        elif final_route == "uncertain":
            final_score -= 0.0
        final_score -= float(record.get("final_score_penalty", 0.0) or 0.0)
        if measured_peak_db < -18.0:
            final_score -= 3.0
        if measured_rms_db < -28.0:
            final_score -= 3.0

        record["final_score"] = float(max(0.0, min(100.0, final_score)))
        record["quality"] = float(max(0.0, min(1.0, record["final_score"] / 100.0)))
        candidates.append(record)

    candidates.sort(
        key=lambda r: (
            float(r.get("final_score", 0.0)),
            extract_speaker_similarity(r),
            float(r.get("duration_sec", 0.0) or 0.0),
        ),
        reverse=True,
    )

    total_seconds = 0.0
    for record in candidates:
        if len(selected) >= max_files:
            break
        speaker_sim = get_speaker_similarity(record)
        if speaker_sim is None:
            continue
        voiceprint_status = str(record.get("voiceprint_status") or record.get("stage_status", {}).get("voiceprint") or "")
        if voiceprint_status not in {"voiceprint_pass", "passed"}:
            final_reject_reasons["voiceprint_not_pass"] += 1
            continue
        metrics = record.get("metrics", {})
        music_like = bool(metrics.get("music_like"))
        singing_like = bool(metrics.get("singing_like") or record.get("singing_like"))
        speaker_count = int(metrics.get("speaker_count", record.get("speaker_count", 1)) or 1)
        overlap_ratio = float(metrics.get("overlap_ratio", record.get("overlap_ratio", 0.0)) or 0.0)
        target_speaker_ratio = float(metrics.get("target_speaker_ratio", record.get("target_speaker_ratio", 1.0)) or 1.0)
        transcript = str(record.get("transcript", "") or "")
        whisper_text_len = len(normalize_match_text(transcript))
        whisper_quality = record.get("metrics", {}).get("whisper_quality")
        whisper_no_speech_prob = metrics.get("whisper_no_speech_prob")
        if music_like:
            final_reject_reasons["music_like"] += 1
            record["final_reasons"].append("music_like")
            continue
        if singing_like:
            final_reject_reasons["singing_like"] += 1
            record["final_reasons"].append("singing_like")
            continue
        if speaker_count > 1:
            final_reject_reasons["multi_speaker"] += 1
            record["final_reasons"].append("multi_speaker")
            continue
        if overlap_ratio > 0.05:
            final_reject_reasons["overlap_speech"] += 1
            record["final_reasons"].append("overlap_speech")
            continue
        if target_speaker_ratio < 0.85:
            final_reject_reasons["target_speaker_ratio_low"] += 1
            record["final_reasons"].append("target_speaker_ratio_low")
            continue
        if whisper_text_len < 2:
            final_reject_reasons["asr_empty"] += 1
            record["final_reasons"].append("asr_empty")
            continue
        if whisper_no_speech_prob is not None and float(whisper_no_speech_prob) > 0.45:
            final_reject_reasons["asr_no_speech_high"] += 1
            record["final_reasons"].append("asr_no_speech_high")
            continue
        if isinstance(whisper_quality, dict) and not whisper_quality.get("accepted", True):
            final_reject_reasons["whisper_quality_rejected"] += 1
            record["final_reasons"].append("whisper_quality_rejected")
            continue
        if speaker_sim < min_speaker_sim or float(record.get("final_score", 0.0)) < 45.0:
            final_reject_reasons["score_too_low"] += 1
            record["final_reasons"].append("score_too_low")
            continue
        selected.append(record)
        total_seconds += float(record.get("duration_sec", 0.0) or 0.0)
        if total_seconds >= target_total_seconds:
            break

    if selected:
        if total_seconds < target_total_seconds:
            LOG.warning(
                "Final total duration below target: %.2fs < %.2fs, keeping %d selected items",
                total_seconds,
                target_total_seconds,
                len(selected),
            )
    else:
        LOG.warning("Final selection produced no items after hard filtering")

    for record in selected:
        record["status"] = "final_selected"
        record["active"] = True
    if selected and total_seconds < target_total_seconds:
        final_reject_reasons["total_duration_not_enough"] += 1
    return selected, final_reject_reasons


def select_final_records_v2(
    records: list[dict[str, Any]],
    max_files: int = FINAL_MAX_FILES,
    target_total_seconds: float = FINAL_TARGET_TOTAL_SECONDS,
    min_speaker_sim: float = MIN_FINAL_SPEAKER_SIM,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    manual_transcribe: list[dict[str, Any]] = []
    final_reject_reasons: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []

    for record in records:
        if not record.get("active", True):
            continue
        record["final_reasons"] = []
        decision, gate_reason = clean_speech_gate(record)
        record["clean_speech_decision"] = decision
        record["clean_speech_reason"] = gate_reason
        if decision == "reject":
            final_reject_reasons[gate_reason] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append(gate_reason)
            continue
        if decision == "manual_transcribe":
            record["quality"] = 0.0
            record["status"] = "manual_transcribe"
            record["active"] = False
            record["final_bucket"] = "manual_transcribe"
            record["final_reasons"].append(gate_reason)
            manual_transcribe.append(record)
            continue

        try:
            duration_sec, measured_rms_db, measured_peak_db, measured_silent_ratio = inspect_final_audio(record)
        except FileNotFoundError:
            final_reject_reasons["file_not_found"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("file_not_found")
            record["note"] = f"missing_path:{record.get('current_path') or record.get('path')}"
            continue
        except Exception as exc:
            final_reject_reasons["cannot_read_audio"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("cannot_read_audio")
            record["note"] = f"cannot_read_audio:{exc}"
            continue

        speaker_sim = get_speaker_similarity(record)
        if speaker_sim is None:
            final_reject_reasons["missing_similarity"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("missing_similarity")
            continue

        metrics = record.get("metrics", {}) or {}
        duration_sec = float(record.get("duration_sec", 0.0) or duration_sec)
        rms_db = float(metrics.get("rms_db", measured_rms_db))
        peak_db = float(metrics.get("peak_dbfs", measured_peak_db))
        silent_ratio_value = float(metrics.get("silent_ratio", measured_silent_ratio))
        music_score = float(record.get("music_score", compute_music_score(record)) or compute_music_score(record))
        asr_text = str(record.get("transcript", "") or "")
        asr_len = len(normalize_match_text(asr_text))
        no_speech_prob = metrics.get("whisper_no_speech_prob")
        vad_ratio = float(metrics.get("vad_ratio", 0.0) or 0.0)
        music_like = bool(metrics.get("music_like"))
        singing_like = bool(metrics.get("singing_like") or record.get("singing_like"))
        num_speakers = int(metrics.get("speaker_count", record.get("speaker_count", 1)) or 1)
        overlap_ratio = float(metrics.get("overlap_ratio", record.get("overlap_ratio", 0.0)) or 0.0)
        target_speaker_ratio = float(metrics.get("target_speaker_ratio", record.get("target_speaker_ratio", 1.0)) or 1.0)
        voiceprint_status = str(record.get("voiceprint_status") or record.get("stage_status", {}).get("voiceprint") or "")
        analysis_status = str(record.get("analysis_status") or record.get("stage_status", {}).get("quality_analyzer") or "")
        pre_b_status = str(record.get("stage_status", {}).get("pre_b") or "")
        demucs_quality = str(record.get("stage_status", {}).get("demucs_quality") or record.get("stage_status", {}).get("demucs") or "")
        route = str(record.get("stage_status", {}).get("vocal_separator") or record.get("stage_status", {}).get("demucs_quality") or "")
        transcript = asr_text
        whisper_quality = metrics.get("whisper_quality")
        negative_margin = record.get("speaker_margin")

        if silent_ratio_value > MAX_SILENT_RATIO:
            final_reject_reasons["too_much_silence"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("too_much_silence")
            continue

        if speaker_sim < min_speaker_sim:
            final_reject_reasons["speaker_similarity_low"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("speaker_similarity_low")
            continue

        final_score = 0.0
        final_score += speaker_sim * 60.0
        if 2.0 <= duration_sec <= 10.0:
            final_score += 15.0
        if -32.0 <= rms_db <= -12.0:
            final_score += 10.0
        if silent_ratio_value < 0.45:
            final_score += 10.0
        if voiceprint_status == "voiceprint_pass":
            final_score += 10.0
        if pre_b_status == "pass_vocals":
            final_score += 4.0
        if demucs_quality == "passed":
            final_score += 3.0
        if analysis_status == "analysis_pass":
            final_score += 2.0
        elif analysis_status == "analysis_uncertain":
            final_score -= 4.0
            record["final_reasons"].append("analysis_uncertain")
        elif analysis_status == "analysis_error_fallback":
            final_score -= 6.0
            record["final_reasons"].append("analysis_error_fallback")
        if route in {"fallback", "uncertain"}:
            final_score -= 2.0
        if not is_low_value_transcript(transcript):
            final_score += 3.0
        if music_score > MAX_MUSIC_SCORE:
            final_reject_reasons["music_score_high"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("music_score_high")
            continue
        if 0.15 < music_score <= MAX_MUSIC_SCORE:
            final_score -= 10.0
            record["final_reasons"].append("music_score_weak")
        if music_like:
            final_score -= 8.0
            record["final_reasons"].append("music_like")
        if singing_like:
            final_reject_reasons["singing_like"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("singing_like")
            continue
        if num_speakers > 1:
            final_reject_reasons["multi_speaker"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("multi_speaker")
            continue
        if overlap_ratio > 0.05:
            final_reject_reasons["overlap_speech"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("overlap_speech")
            continue
        if target_speaker_ratio < 0.85:
            final_reject_reasons["target_speaker_ratio_low"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("target_speaker_ratio_low")
            continue
        if asr_len < MIN_ASR_TEXT_LEN:
            if no_speech_prob is not None and float(no_speech_prob) >= 0.60:
                final_reject_reasons["asr_empty_high_no_speech"] += 1
                record["quality"] = 0.0
                record["status"] = "final_rejected"
                record["active"] = False
                record["final_bucket"] = "reject"
                record["final_reasons"].append("asr_empty_high_no_speech")
                continue
            if speaker_sim >= TARGET_SIM_TH and vad_ratio >= MIN_VAD_RATIO and music_score <= MAX_MUSIC_SCORE:
                record["quality"] = 0.0
                record["status"] = "manual_transcribe"
                record["active"] = False
                record["final_bucket"] = "manual_transcribe"
                record["final_reasons"].append("asr_empty_but_voiceprint_high")
                manual_transcribe.append(record)
                continue
            final_reject_reasons["asr_empty"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("asr_empty")
            continue
        if singing_like:
            final_reject_reasons["singing_like"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("singing_like")
            continue
        if negative_margin is not None and float(negative_margin) < NEGATIVE_MARGIN_TH:
            final_reject_reasons["negative_margin_low"] += 1
            record["quality"] = 0.0
            record["status"] = "final_rejected"
            record["active"] = False
            record["final_bucket"] = "reject"
            record["final_reasons"].append("negative_margin_low")
            continue

        if float(record.get("final_score_penalty", 0.0) or 0.0):
            final_score -= float(record.get("final_score_penalty", 0.0) or 0.0)
        if peak_db < -18.0:
            final_score -= 3.0
        if rms_db < -28.0:
            final_score -= 3.0

        record["final_score"] = float(max(0.0, min(100.0, final_score)))
        record["quality"] = float(max(0.0, min(1.0, record["final_score"] / 100.0)))
        candidates.append(record)

    candidates.sort(
        key=lambda r: (
            float(r.get("final_score", 0.0)),
            extract_speaker_similarity(r),
            float(r.get("duration_sec", 0.0) or 0.0),
        ),
        reverse=True,
    )

    total_seconds = 0.0
    for record in candidates:
        if len(selected) >= max_files:
            break
        speaker_sim = get_speaker_similarity(record)
        if speaker_sim is None:
            continue
        if float(record.get("final_score", 0.0)) < 45.0:
            final_reject_reasons["score_too_low"] += 1
            record["final_reasons"].append("score_too_low")
            continue
        selected.append(record)
        record["final_bucket"] = "final_selected"
        total_seconds += float(record.get("duration_sec", 0.0) or 0.0)
        if total_seconds >= target_total_seconds:
            break

    if selected and total_seconds < target_total_seconds:
        LOG.warning(
            "Final total duration below target: %.2fs < %.2fs, keeping %d selected items",
            total_seconds,
            target_total_seconds,
            len(selected),
        )
    elif not selected:
        LOG.warning("Final selection produced no items after hard filtering")

    for record in selected:
        record["status"] = "final_selected"
        record["active"] = True
    if selected and total_seconds < target_total_seconds:
        final_reject_reasons["total_duration_not_enough"] += 1
    return selected, manual_transcribe, final_reject_reasons


def export_quality_buckets(
    final_selected: list[dict[str, Any]],
    manual_transcribe: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    dirs: dict[str, Path],
    transcripts: dict[str, str],
) -> None:
    def _export_bucket(items: list[dict[str, Any]], bucket_wavs: Path, bucket_texts: Path, bucket_name: str) -> list[dict[str, Any]]:
        ensure_dir(bucket_wavs)
        ensure_dir(bucket_texts)
        manifest_rows: list[dict[str, Any]] = []
        for idx, record in enumerate(items, 1):
            src = Path(record.get("current_path") or record.get("path_for_verify") or record.get("path_for_vad") or "")
            if not src.exists():
                continue
            sample_id = str(record.get("sample_id") or src.stem)
            dst = bucket_wavs / f"{idx:04d}_{src.name}"
            copy_or_move(src, dst, move=False)
            text = transcripts.get(sample_id, str(record.get("transcript", "") or ""))
            (bucket_texts / f"{idx:04d}_{src.stem}.txt").write_text(text, encoding="utf-8")
            row = {
                "segment_id": record.get("segment_id"),
                "sample_id": sample_id,
                "path": str(dst.resolve().as_posix()),
                "bucket": bucket_name,
                "speaker_similarity": record.get("speaker_similarity"),
                "music_score": compute_music_score(record),
                "silent_ratio": record.get("metrics", {}).get("silent_ratio"),
                "vad_ratio": record.get("metrics", {}).get("vad_ratio"),
                "asr_text": text,
                "asr_text_len": len(normalize_match_text(text)),
                "analysis_status": record.get("analysis_status"),
                "final_bucket": record.get("final_bucket"),
                "final_reason": record.get("clean_speech_reason") or ";".join(record.get("final_reasons") or []),
                "final_score": record.get("final_score"),
            }
            manifest_rows.append(row)
        manifest_path = bucket_wavs.parent / "manifest.jsonl"
        write_jsonl(manifest_rows, manifest_path)
        csv_path = bucket_wavs.parent / "manifest.csv"
        if manifest_rows:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)
        else:
            csv_path.write_text("", encoding="utf-8")
        return manifest_rows

    _export_bucket(final_selected, dirs["final_selected_wavs"], dirs["final_selected_texts"], "final_selected")
    _export_bucket(manual_transcribe, dirs["manual_transcribe_wavs"], dirs["manual_transcribe_texts"], "manual_transcribe")
    _export_bucket(rejected, dirs["reject_wavs"], dirs["reject_texts"], "reject")


def export_final_dataset(
    selected: list[dict[str, Any]],
    final_dir: Path,
    dataset_root: Path,
    transcripts: dict[str, str],
    train_ratio: float = 0.9,
) -> dict[str, int]:
    ensure_dir(final_dir)
    dataset_dirs = build_dataset_dirs(dataset_root)
    final_wavs = final_dir / "wavs"
    final_texts = final_dir / "transcripts"
    ensure_dir(final_wavs)
    ensure_dir(final_texts)

    wav_paths: list[Path] = []
    for record in selected:
        src = Path(record["current_path"])
        sample_id = str(record.get("sample_id") or src.stem)
        wav_dst = final_wavs / f"{sample_id}.wav"
        copy_or_move(src, wav_dst, move=False)
        txt = transcripts.get(sample_id, "")
        (final_texts / f"{sample_id}.txt").write_text(txt, encoding="utf-8")
        wav_paths.append(wav_dst)
        record["stage_paths"]["final"] = str(wav_dst)
        record["current_path"] = str(wav_dst)
        record["stage_status"]["final"] = "passed"

    random.shuffle(wav_paths)
    split_idx = int(len(wav_paths) * train_ratio)
    train = wav_paths[:split_idx]
    val = wav_paths[split_idx:]

    for item in train:
        copy_or_move(item, dataset_dirs["train_wavs"] / item.name, move=False)
        txt = transcripts.get(item.stem, "")
        (dataset_dirs["train_texts"] / f"{item.stem}.txt").write_text(txt, encoding="utf-8")
    for item in val:
        copy_or_move(item, dataset_dirs["val_wavs"] / item.name, move=False)
        txt = transcripts.get(item.stem, "")
        (dataset_dirs["val_texts"] / f"{item.stem}.txt").write_text(txt, encoding="utf-8")

    return {"train": len(train), "val": len(val)}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        summary[status] = summary.get(status, 0) + 1
    return summary


def run_augmentations(selected: list[dict[str, Any]], aug_dir: Path, limit: int = 20) -> None:
    if not selected:
        return
    try:
        import librosa
        import soundfile as sf

        ensure_dir(aug_dir)
        for record in selected[: min(limit, len(selected))]:
            src = Path(record["current_path"])
            y, sr = librosa.load(str(src), sr=16000)
            for steps in (-1.0, 1.0):
                y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
                out_path = aug_dir / f"{src.stem}_pitch{int(steps)}.wav"
                sf.write(str(out_path), y2, sr, subtype="PCM_16")
    except Exception as exc:
        LOG.warning("augmentation failed: %s", exc)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cleaning pipeline v2 with manifest and pluggable stages")
    parser.add_argument("--input-dir", "-i", type=str, default=None, help="原始音频目录")
    parser.add_argument("--out-root", type=str, default=str(Path(CFG.get("temp_dir") or "./pipeline_temp") / "clean_v2"))
    parser.add_argument("--run-version", type=str, default=None, help="运行版本号 / 实验标签")
    parser.add_argument("--resume-manifest", type=str, default=None, help="从已有 manifest.jsonl 继续运行")
    parser.add_argument("--dataset-root", type=str, default=None, help="最终 dataset 输出路径")
    parser.add_argument("--speaker-ref", type=str, default=None, help="声纹参考音频文件或目录")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repair-rounds", type=int, default=2, help="???????")
    parser.add_argument("--voiceprint-batch-size", type=int, default=None, help="声纹比对批大小，默认沿用 --batch-size")
    parser.add_argument("--pyannote-batch-size", type=int, default=1, help="pyannote 批大小")
    parser.add_argument("--whisper-batch-size", type=int, default=2, help="Whisper 批大小")
    parser.add_argument("--demucs-batch-size", type=int, default=1, help="Demucs 批大小")
    parser.add_argument("--dedupe-thresh", type=float, default=0.985)
    parser.add_argument("--quality-thres", type=float, default=0.6)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--final-max-files", type=int, default=FINAL_MAX_FILES, help="final 阶段最多保留的文件数")
    parser.add_argument("--final-target-total-seconds", type=float, default=FINAL_TARGET_TOTAL_SECONDS, help="final 阶段目标总时长（秒）")
    parser.add_argument("--final-min-speaker-sim", type=float, default=MIN_FINAL_SPEAKER_SIM, help="final 阶段声纹最小硬门槛")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--skip-demucs", action="store_true", help="兼容旧参数：跳过 Demucs")
    parser.add_argument("--skip-whisper", action="store_true", help="兼容旧参数：跳过 Whisper")
    parser.add_argument("--skip-pyannote", action="store_true", help="跳过 pyannote")
    parser.add_argument("--skip-voiceprint", action="store_true", help="跳过声纹比对")
    parser.add_argument("--enable-demucs", action="store_true", help="启用 Demucs")
    parser.add_argument("--enable-whisper", action="store_true", help="启用 Whisper 转录")
    parser.add_argument("--enable-pyannote", action="store_true", help="启用 pyannote 说话人分离")
    parser.add_argument("--enable-voiceprint", action="store_true", help="启用声纹比对")
    parser.add_argument("--legacy-flow", action="store_true", help="?????? stage ??")
    parser.add_argument("--min-seg-sec", type=float, default=3.0)
    parser.add_argument("--max-seg-sec", type=float, default=10.0)
    parser.add_argument("--min-rms-db", type=float, default=-50.0)
    parser.add_argument("--max-clip-ratio", type=float, default=0.02)
    parser.add_argument("--min-snr-db", type=float, default=5.0)
    parser.add_argument("--max-reverb-score", type=float, default=0.90)
    parser.add_argument("--min-vad-ratio", type=float, default=MIN_SPEECH_RATIO)
    parser.add_argument("--speaker-thres", type=float, default=0.68)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--filter-multi-speaker", action="store_true", default=True)
    parser.add_argument("--allow-multi-speaker", action="store_true", help="允许多说话人片段通过")
    return parser


def resolve_stage_flags(args: argparse.Namespace) -> dict[str, bool]:
    enable_demucs = bool(args.enable_demucs and not args.skip_demucs)
    enable_whisper = bool(args.enable_whisper and not args.skip_whisper)
    enable_pyannote = bool(args.enable_pyannote and not args.skip_pyannote)
    enable_voiceprint = bool(args.enable_voiceprint and not args.skip_voiceprint)
    filter_multi = not bool(args.allow_multi_speaker)
    return {
        "demucs": enable_demucs,
        "whisper": enable_whisper,
        "pyannote": enable_pyannote,
        "voiceprint": enable_voiceprint,
        "filter_multi_speaker": filter_multi,
    }


def dry_run_summary(files: list[Path], args: argparse.Namespace, flags: dict[str, bool]) -> None:
    LOG.info("DRY RUN: 将执行以下步骤：")
    LOG.info("  1. 格式转换 (16k/mono WAV)")
    LOG.info("  2. 切片 (%.1f~%.1fs) + 去重", args.min_seg_sec, args.max_seg_sec)
    LOG.info("  3. Pre-A: 音量/削波 (rms_db>=%.1f, clip<=%.3f)", args.min_rms_db, args.max_clip_ratio)
    LOG.info("  4. Demucs: %s", "启用" if flags["demucs"] else "跳过")
    LOG.info("  5. Pre-B: SNR/混响 (snr_db>=%.1f, reverb<=%.2f)", args.min_snr_db, args.max_reverb_score)
    LOG.info("  6. VAD (min_ratio=%.2f)", args.min_vad_ratio)
    LOG.info("  7. pyannote: %s", "启用" if flags["pyannote"] else "跳过")
    LOG.info("  8. 声纹比对: %s", "启用" if flags["voiceprint"] else "跳过")
    LOG.info("  9. Whisper: %s", "启用" if flags["whisper"] else "跳过")
    LOG.info(" 10. Post 质量打分 + 标准化数据集输出")
    LOG.info("输入文件数: %d", len(files))



def dry_run_modular_summary(files: list[Path], args: argparse.Namespace, flags: dict[str, bool]) -> None:
    LOG.info("DRY RUN: ?????????????")
    LOG.info("  1. AudioLoader: ?? / ????? / ????")
    LOG.info("  2. QualityAnalyzer: RMS / LUFS / peak / silent_ratio / SNR / music / reverb")
    LOG.info("  3. Router: ??? LoudnessFixer / VocalSeparator / SpeechDenoiser / DereverbModule / VADSegmenter")
    LOG.info("  4. SpeakerVerifier: ????")
    LOG.info("  5. WhisperValidator: ???????")
    LOG.info("  6. FinalScorer: pass / uncertain / reject")
    LOG.info("????: demucs=%s, pyannote=%s, voiceprint=%s, whisper=%s", flags["demucs"], flags["pyannote"], flags["voiceprint"], flags["whisper"])
    LOG.info("?????: %d", len(files))

def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    flags = resolve_stage_flags(args)
    run_version = make_run_version(args.run_version)

    out_root = Path(args.out_root)
    ensure_dir(out_root)
    dirs = build_stage_dirs(out_root)
    dataset_root = Path(args.dataset_root) if args.dataset_root else dirs["dataset"]
    ensure_dir(dataset_root)

    if not getattr(args, "legacy_flow", False):
        return run_modular_pipeline(args, flags, run_version, out_root, dirs, dataset_root)

    raw_src_dir = Path(args.input_dir or Path(CFG.get("temp_dir") or "./pipeline_temp") / "raw")
    resume_mode = bool(args.resume_manifest)
    all_files: list[Path] = []
    source_entries: list[dict[str, Any]] = []
    convertible: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    active_records: list[dict[str, Any]] = []

    if resume_mode:
        resume_path = Path(args.resume_manifest)
        if not resume_path.exists():
            LOG.error("resume manifest does not exist: %s", resume_path)
            return 1
        active_records = load_manifest_records(resume_path)
        segment_records = list(active_records)
        LOG.info("Loaded %d records from resume manifest %s", len(active_records), resume_path)
        if args.dry_run:
            LOG.info("DRY RUN: resume mode would continue from the loaded manifest through the advanced stages.")
            return 0
    else:
        if not raw_src_dir.exists():
            LOG.error("输入目录不存在: %s", raw_src_dir)
            return 1

        all_files = split_source_files(raw_src_dir, start_index=args.start_index, max_files=args.max_files)
        LOG.info("Found %d raw files in %s (processing %d)", len(all_files), raw_src_dir, len(all_files))

        if args.dry_run:
            dry_run_summary(all_files, args, flags)
            return 0

        for source_path in all_files:
            try:
                source_entries.append(make_source_entry(source_path, dirs["raw"], move=args.move))
            except Exception as exc:
                LOG.warning("copy/move source failed %s: %s", source_path, exc)

        LOG.info("Converted stage will process %d source files", len(source_entries))

        for entry in source_entries:
            try:
                converted_path = dirs["converted"] / f"{Path(entry['raw_path']).stem}_16k.wav"
                convert_to_wav16_mono(Path(entry["raw_path"]), converted_path)
                entry["converted_path"] = str(converted_path)
            except Exception as exc:
                LOG.warning("转换失败 %s: %s", entry["raw_path"], exc)
                entry["converted_path"] = None

        convertible = [entry for entry in source_entries if entry.get("converted_path")]
        LOG.info("Converted %d files", len(convertible))

        for entry in convertible:
            try:
                segment_records.extend(
                    segment_source_file(
                        entry,
                        dirs["segments"],
                        min_seg_sec=args.min_seg_sec,
                        max_seg_sec=args.max_seg_sec,
                    )
                )
            except Exception as exc:
                LOG.warning("切片失败 %s: %s", entry["converted_path"], exc)

        LOG.info("Created %d segments", len(segment_records))

        active_records = simple_dedupe_by_mfcc(segment_records, thresh=args.dedupe_thresh)
        LOG.info("After dedupe: %d segments", len(active_records))

        active_records = run_pre_a(
            active_records,
            dirs["pre_a"],
            min_rms_db=args.min_rms_db,
            max_clip_ratio=args.max_clip_ratio,
            move=args.move,
        )
        LOG.info("Pre-A pass: %d", len(active_records))

    for record in segment_records:
        record["run_version"] = run_version

    LOG.info("Demucs stage start: %d items, batch=%d, device=%s", len(active_records), max(1, int(args.demucs_batch_size)), "auto")
    active_records = run_demucs_stage(
        active_records,
        dirs["demucs"],
        enabled=flags["demucs"],
        device="auto",
        batch_size=max(1, int(args.demucs_batch_size)),
        attempts_dir=dirs["demucs_attempts"],
    )

    LOG.info("Demucs quality gate start: %d items", len(active_records))
    active_records = run_demucs_quality_gate(
        active_records,
        enabled_whisper_probe=flags["whisper"],
        whisper_batch_size=max(1, int(args.whisper_batch_size)),
        uncertain_dir=dirs["review_uncertain"],
    )
    LOG.info("Demucs quality pass: %d", len(active_records))

    active_records = run_pre_b(
        active_records,
        dirs["pre_b"],
        min_snr_db=args.min_snr_db,
        max_reverb_score=args.max_reverb_score,
        move=args.move,
    )
    LOG.info("Pre-B pass: %d", len(active_records))

    vad_stats = Counter()
    LOG.info("VAD stage start: %d items, min_ratio=%.2f", len(active_records), float(args.min_vad_ratio))
    active_records = run_vad_stage(active_records, dirs["vad"], min_vad_ratio=float(args.min_vad_ratio), move=args.move, stats=vad_stats)
    LOG.info("VAD pass: %d", len(active_records))
    LOG.info("VAD stats: %s", dict(vad_stats))
    if vad_stats.get("vad_input", 0) > 0 and vad_stats.get("vad_pass", 0) == 0:
        LOG.warning("VAD passed 0 items while inputs were available; threshold may still be too strict.")

    LOG.info("Pyannote stage start: %d items, batch=%d, device=%s", len(active_records), max(1, int(args.pyannote_batch_size)), "auto")
    active_records = run_pyannote_stage(
        active_records,
        dirs["diarized"],
        enabled=flags["pyannote"],
        filter_multi_speaker=flags["filter_multi_speaker"],
        batch_size=max(1, int(args.pyannote_batch_size)),
        device="auto",
    )
    LOG.info("Diarization stage items: %d", len(active_records))

    LOG.info("Voiceprint stage start: %d items, batch=%d", len(active_records), max(1, int(args.voiceprint_batch_size or args.batch_size)))
    active_records, voiceprint_stats = run_voiceprint_stage(
        active_records,
        dirs["verified"],
        speaker_ref=args.speaker_ref,
        enabled=flags["voiceprint"],
        speaker_thres=args.speaker_thres,
        batch_size=max(1, int(args.voiceprint_batch_size or args.batch_size)),
        model_path=CFG.get("mb_encoder"),
        move=args.move,
    )
    LOG.info("Verified speakers: %d", len(active_records))

    if flags["whisper"]:
        LOG.info("Whisper stage start: %d items, batch=%d", len(active_records), max(1, int(args.whisper_batch_size)))
        transcripts = transcribe_with_whisper(active_records, batch_size=max(1, int(args.whisper_batch_size)))
    else:
        transcripts = {Path(record["current_path"]).name: "" for record in active_records}
        for record in active_records:
            record["stage_status"]["whisper"] = "skipped"

    verified_paths = [Path(record["current_path"]) for record in active_records]
    for record in active_records:
        sample_id = str(record.get("sample_id") or Path(record["current_path"]).stem)
        txt = transcripts.get(sample_id, "")
        record["transcript"] = txt
        record["transcript_conf"] = compute_transcript_confidence(txt, record.get("metrics")) if txt else 0.0
        if flags["whisper"]:
            record["stage_status"]["whisper"] = "passed"
        txt_path = dirs["final_texts"] / f"{sample_id}.txt"
        ensure_dir(txt_path.parent)
        txt_path.write_text(txt, encoding="utf-8")

    final_candidates = len([r for r in active_records if r.get("active", True)])
    selected, manual_transcribe, final_reject_reasons = select_final_records_v2(
        active_records,
        max_files=max(1, int(getattr(args, "final_max_files", FINAL_MAX_FILES) or FINAL_MAX_FILES)),
        target_total_seconds=float(getattr(args, "final_target_total_seconds", FINAL_TARGET_TOTAL_SECONDS)),
        min_speaker_sim=float(getattr(args, "final_min_speaker_sim", MIN_FINAL_SPEAKER_SIM)),
    )
    LOG.info("Final selected count: %d", len(selected))
    LOG.info("Manual transcribe count: %d", len(manual_transcribe))
    LOG.info("final_reject_reasons:")
    for reason, count in final_reject_reasons.most_common():
        LOG.info("  %s: %d", reason, count)

    rejected_items = [record for record in active_records if str(record.get("final_bucket") or "") == "reject"]
    export_quality_buckets(selected, manual_transcribe, rejected_items, dirs, transcripts)
    LOG.info("Final buckets exported: final_selected=%d manual_transcribe=%d reject=%d", len(selected), len(manual_transcribe), len(rejected_items))

    resume_manifest_path = out_root / "resume_manifest.jsonl"
    write_jsonl(selected, resume_manifest_path)

    if args.augment and selected:
        run_augmentations(selected, dirs["aug"])

    train_stats = export_final_dataset(
        selected,
        dirs["final"],
        dataset_root,
        transcripts,
        train_ratio=args.train_ratio,
    )

    manifest_records = [record for record in segment_records]
    write_jsonl(manifest_records, dirs["manifest"])
    try:
        if dataset_root != dirs["dataset"]:
            write_jsonl(selected, dataset_root / "manifest.jsonl")
        else:
            write_jsonl(selected, dataset_root / "manifest.jsonl")
    except Exception as exc:
        LOG.warning("Failed to write dataset manifest: %s", exc)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "run_config": {
            "input_dir": str(raw_src_dir),
            "out_root": str(out_root),
            "dataset_root": str(dataset_root),
            "speaker_ref": args.speaker_ref,
            "run_version": run_version,
            "stage_flags": flags,
            "batch_sizes": {
                "demucs": max(1, int(args.demucs_batch_size)),
                "pyannote": max(1, int(args.pyannote_batch_size)),
                "voiceprint": max(1, int(args.voiceprint_batch_size or args.batch_size)),
                "whisper": max(1, int(args.whisper_batch_size)),
            },
            "quality_thres": args.quality_thres,
            "dedupe_thres": args.dedupe_thresh,
            "final_max_files": max(1, int(args.final_max_files or FINAL_MAX_FILES)),
            "final_target_total_seconds": float(args.final_target_total_seconds),
            "final_min_speaker_sim": float(args.final_min_speaker_sim),
            "quality_weights": QUALITY_WEIGHTS,
        },
        "counts": {
            "raw_total": len(all_files),
            "processed": len(source_entries),
            "converted": len(convertible),
            "segments": len(segment_records),
            "unique_segments": len([r for r in segment_records if r.get("status") != "dropped_duplicate"]),
            "pre_a_pass": len([r for r in segment_records if r.get("stage_status", {}).get("pre_a") == "passed"]),
            "demucs_attempted": len([r for r in segment_records if r.get("stage_status", {}).get("demucs") in ("passed", "failed", "skipped_duration")]),
            "demucs_success": len([r for r in segment_records if r.get("stage_status", {}).get("demucs") == "passed"]),
            "demucs_failed": len([r for r in segment_records if r.get("stage_status", {}).get("demucs") == "failed"]),
            "pre_b_pass_vocals": len([r for r in segment_records if r.get("stage_status", {}).get("pre_b") == "pass_vocals"]),
            "pre_b_fallback_original": len([r for r in segment_records if r.get("stage_status", {}).get("pre_b") == "fallback_original"]),
            "pre_b_uncertain": len([r for r in segment_records if r.get("stage_status", {}).get("pre_b") == "uncertain"]),
            "pre_b_reject": len([r for r in segment_records if r.get("status") == "dropped_pre_b"]),
            "vad_input": int(vad_stats.get("vad_input", 0)),
            "vad_pass": int(vad_stats.get("vad_pass", 0)),
            "vad_reject_no_speech": int(vad_stats.get("vad_reject_no_speech", 0)),
            "vad_reject_too_short": int(vad_stats.get("vad_reject_too_short", 0)),
            "vad_reject_read_error": int(vad_stats.get("vad_reject_read_error", 0)),
            "diarized_pass": len([r for r in segment_records if r.get("stage_status", {}).get("pyannote") in ("passed", "skipped", "unavailable")]),
            "verified_input": len([r for r in segment_records if r.get("stage_status", {}).get("vad") == "passed"]),
            "voiceprint_input": int(voiceprint_stats.get("voiceprint_input", 0)),
            "voiceprint_cli_status": voiceprint_stats.get("voiceprint_cli_status", ""),
            "voiceprint_result_count": int(voiceprint_stats.get("voiceprint_result_count", 0)),
            "voiceprint_written": int(voiceprint_stats.get("voiceprint_written", 0)),
            "voiceprint_compute_error": int(voiceprint_stats.get("voiceprint_compute_error", 0)),
            "voiceprint_missing_result": int(voiceprint_stats.get("voiceprint_missing_result", 0)),
            "voiceprint_missing_similarity": int(voiceprint_stats.get("voiceprint_missing_similarity", 0)),
            "verified_pass": int(voiceprint_stats.get("verified_pass", 0)),
            "verified_uncertain": int(voiceprint_stats.get("verified_uncertain", 0)),
            "verified_reject": int(voiceprint_stats.get("verified_reject", 0)),
            "final_candidates": final_candidates,
            "final_selected": len(selected),
            "train": train_stats.get("train", 0),
            "val": train_stats.get("val", 0),
        },
        "status_breakdown": summarize_records(segment_records),
        "output": {
            "manifest": str(dirs["manifest"]),
            "report": str(dirs["report"]),
            "final_wavs": str(dirs["final_wavs"]),
            "final_texts": str(dirs["final_texts"]),
            "dataset_root": str(dataset_root),
        },
        "final_reject_reasons": dict(final_reject_reasons),
        "final_selected": [
            {
                "sample_id": record["sample_id"],
                "path": record["current_path"],
                "quality": record.get("quality", 0.0),
                "final_score": record.get("final_score", 0.0),
                "final_route": record.get("final_route"),
                "analysis_status": record.get("analysis_status"),
                "analysis_score": record.get("analysis_score", 0.0),
                "speaker_similarity": record.get("speaker_similarity"),
                "voiceprint_status": record.get("voiceprint_status"),
                "voiceprint_target_key": record.get("voiceprint_target_key"),
                "snr_db": record["metrics"].get("snr_db"),
                "vad_ratio": record["metrics"].get("vad_ratio"),
                "transcript_conf": record.get("transcript_conf", 0.0),
                "transcript": record.get("transcript", ""),
                "run_version": record.get("run_version", run_version),
            }
            for record in selected[:200]
        ],
    }

    dirs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if dataset_root != dirs["dataset"]:
        (dataset_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        (dataset_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info("Pipeline finished. Report: %s", dirs["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
