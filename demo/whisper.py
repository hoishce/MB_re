from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

import torch


def _ensure_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_repo_root()

try:
    import ensure_dlls

    ensure_dlls.ensure_dll_priority()
except Exception:
    pass


def _check_numpy_compatibility() -> None:
    try:
        numpy_version = package_version("numpy").split("+", 1)[0]
    except PackageNotFoundError as exc:
        raise RuntimeError("NumPy is required. Please install project dependencies first.") from exc

    profile = os.environ.get("MOCKINGBIRD_PIPELINE_PROFILE", "core").strip().lower() or "core"
    parts = numpy_version.split(".")
    version_tuple = tuple(int(x) if x.isdigit() else 0 for x in (parts + ["0", "0"])[:3])
    if profile == "advanced":
        if version_tuple < (2, 2, 2):
            raise RuntimeError(
                f"Detected NumPy {numpy_version}, but the advanced profile expects NumPy 2.2.2 or newer."
            )
    elif numpy_version != "1.26.4":
        raise RuntimeError(f"Detected NumPy {numpy_version}, but this project expects NumPy 1.26.4.")


_check_numpy_compatibility()


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_CUDA_DLL_DIR = os.environ.get("MOCKINGBIRD_CUDA_DLL_DIR")
if _CUDA_DLL_DIR and os.path.exists(_CUDA_DLL_DIR):
    os.add_dll_directory(_CUDA_DLL_DIR)

from faster_whisper import WhisperModel


def _pick_whisper_device() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


def _resolve_model_spec(model_spec: str | None) -> tuple[str, bool]:
    if not model_spec:
        model_spec = os.environ.get("MOCKINGBIRD_WHISPER_MODEL", "large-v3")
    path = Path(model_spec)
    if path.exists():
        return str(path), True
    return model_spec, False


_WHISPER_MODEL: WhisperModel | None = None
_WHISPER_MODEL_KEY: tuple[str, str, str, int, bool] | None = None


def load_whisper_model(
    model_spec: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    device_index: int | None = None,
    local_files_only: bool | None = None,
) -> WhisperModel:
    global _WHISPER_MODEL, _WHISPER_MODEL_KEY

    resolved_model, is_local_path = _resolve_model_spec(model_spec)
    resolved_device, resolved_compute = _pick_whisper_device()
    if device:
        resolved_device = device
        if not compute_type:
            resolved_compute = "float16" if resolved_device == "cuda" else "float32"
    if compute_type:
        resolved_compute = compute_type
    resolved_index = 0 if device_index is None else int(device_index)
    resolved_local_only = is_local_path if local_files_only is None else bool(local_files_only)
    key = (resolved_model, resolved_device, resolved_compute, resolved_index, resolved_local_only)
    if _WHISPER_MODEL is not None and _WHISPER_MODEL_KEY == key:
        return _WHISPER_MODEL

    print(f"Loading Whisper model: {resolved_model} [{resolved_device}/{resolved_compute}]")
    _WHISPER_MODEL = WhisperModel(
        resolved_model,
        device=resolved_device,
        device_index=resolved_index if resolved_device == "cuda" else None,
        compute_type=resolved_compute,
        local_files_only=resolved_local_only,
    )
    _WHISPER_MODEL_KEY = key
    return _WHISPER_MODEL


def _audio_duration_sec(audio_path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        return 0.0


def _normalize_text(text: str) -> str:
    return "".join(str(text).split()).lower()


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def analyze_transcription_quality(
    text: str,
    audio_duration_sec: float,
    transcript_duration_sec: float,
    avg_logprob: float | None,
    no_speech_prob: float | None,
) -> dict[str, Any]:
    clean = _normalize_text(text)
    reasons: list[str] = []
    if not clean:
        reasons.append("empty")
    if len(clean) < 4:
        reasons.append("too_short")
    if not _contains_chinese(clean):
        reasons.append("no_chinese")
    if no_speech_prob is not None and no_speech_prob > 0.55:
        reasons.append("high_no_speech_prob")
    if avg_logprob is not None and avg_logprob < -1.2:
        reasons.append("low_logprob")
    if audio_duration_sec > 0:
        ratio = transcript_duration_sec / max(audio_duration_sec, 1e-6)
        if ratio < 0.25 or ratio > 1.50:
            reasons.append("duration_mismatch")
    confidence = min(1.0, len(clean) / 18.0)
    if avg_logprob is not None:
        confidence *= max(0.0, min(1.0, (avg_logprob + 1.5) / 1.5))
    if no_speech_prob is not None:
        confidence *= max(0.0, 1.0 - float(no_speech_prob))
    accepted = not reasons
    uncertain = accepted and (confidence < 0.55 or len(clean) < 10 or (no_speech_prob is not None and no_speech_prob > 0.25))
    return {
        "accepted": bool(accepted),
        "uncertain": bool(uncertain),
        "reasons": reasons,
        "confidence": float(max(0.0, min(1.0, confidence))),
    }


def transcribe_with_stats(
    audio_path: Path,
    model: WhisperModel,
    beam_size: int = 1,
    vad_filter: bool = True,
    language: str | None = "zh",
) -> dict[str, Any]:
    audio_path = Path(audio_path)
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        vad_filter=vad_filter,
        language=language,
    )

    text_parts: list[str] = []
    segment_count = 0
    transcript_duration_sec = 0.0
    avg_logprob_values: list[float] = []
    no_speech_values: list[float] = []
    segment_rows: list[dict[str, Any]] = []

    for segment in segments:
        segment_count += 1
        seg_text = str(getattr(segment, "text", "") or "").strip()
        if seg_text:
            text_parts.append(seg_text)
        start = float(getattr(segment, "start", 0.0) or 0.0)
        end = float(getattr(segment, "end", start) or start)
        transcript_duration_sec += max(0.0, end - start)
        avg_lp = getattr(segment, "avg_logprob", None)
        no_speech = getattr(segment, "no_speech_prob", None)
        if avg_lp is not None:
            avg_logprob_values.append(float(avg_lp))
        if no_speech is not None:
            no_speech_values.append(float(no_speech))
        segment_rows.append(
            {
                "start": start,
                "end": end,
                "text": seg_text,
                "avg_logprob": None if avg_lp is None else float(avg_lp),
                "no_speech_prob": None if no_speech is None else float(no_speech),
            }
        )

    text = " ".join(text_parts).strip()
    audio_duration_sec = float(getattr(info, "duration", 0.0) or 0.0) or _audio_duration_sec(audio_path)
    avg_logprob = float(sum(avg_logprob_values) / len(avg_logprob_values)) if avg_logprob_values else None
    no_speech_prob = float(sum(no_speech_values) / len(no_speech_values)) if no_speech_values else None
    quality = analyze_transcription_quality(text, audio_duration_sec, transcript_duration_sec, avg_logprob, no_speech_prob)
    return {
        "text": text,
        "audio_duration_sec": audio_duration_sec,
        "transcript_duration_sec": transcript_duration_sec,
        "segment_count": segment_count,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "quality": quality,
        "segments": segment_rows,
    }


def transcribe_one(
    audio_path: Path,
    model: WhisperModel,
    beam_size: int = 1,
    vad_filter: bool = True,
    language: str | None = "zh",
) -> str:
    return str(transcribe_with_stats(audio_path, model, beam_size=beam_size, vad_filter=vad_filter, language=language)["text"])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio with faster-whisper.")
    parser.add_argument("--audio", required=True, help="Path to input audio")
    parser.add_argument("--model", default=os.environ.get("MOCKINGBIRD_WHISPER_MODEL", "large-v3"), help="Model name or local model path")
    parser.add_argument("--output", default=None, help="Output transcript path (.txt). Defaults to <audio stem>.txt")
    parser.add_argument("--device", default=None, choices=("cuda", "cpu"), help="Whisper device override")
    parser.add_argument("--compute-type", default=None, help="Whisper compute type override")
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device index")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument("--language", default="zh", help="Language hint")
    parser.add_argument("--vad-filter", action="store_true", default=True, help="Enable VAD filtering")
    parser.add_argument("--no-vad-filter", action="store_false", dest="vad_filter", help="Disable VAD filtering")
    parser.add_argument("--local-files-only", action="store_true", help="Only load local model files")
    parser.add_argument("--stats-json", default=None, help="Optional JSON file to store transcription stats")
    parser.add_argument("--print-stats", action="store_true", help="Print transcription stats to stdout")
    return parser


def run_cli() -> int:
    args = _build_arg_parser().parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return 1

    model = load_whisper_model(
        model_spec=args.model,
        device=args.device,
        compute_type=args.compute_type,
        device_index=args.device_index,
        local_files_only=args.local_files_only,
    )

    start_time = time.time()
    stats = transcribe_with_stats(
        audio_path,
        model,
        beam_size=max(1, int(args.beam_size)),
        vad_filter=bool(args.vad_filter),
        language=args.language or None,
    )
    text = str(stats.get("text", "") or "")
    output_path = Path(args.output).expanduser().resolve() if args.output else audio_path.with_suffix(".txt")
    output_path.write_text(text, encoding="utf-8")

    if args.stats_json:
        Path(args.stats_json).expanduser().resolve().write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.print_stats:
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    elapsed = time.time() - start_time
    print(f"Transcription written to: {output_path}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Quality: {json.dumps(stats.get('quality', {}), ensure_ascii=False)}")
    return 0 if stats.get("quality", {}).get("accepted", bool(text)) else 2


if __name__ == "__main__":
    raise SystemExit(run_cli())
