#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any


def load_pipeline_module():
    here = Path(__file__).resolve().parent
    pipeline_py = here / "pipeline.py"
    repo_root = here.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import ensure_dlls

        ensure_dlls.ensure_dll_priority()
    except Exception:
        pass
    spec = importlib.util.spec_from_file_location("demo_pipeline", str(pipeline_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module: {pipeline_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonicalize_url(url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return f"youtube.com/watch?v={parsed.path.lstrip('/')}"
    if "youtube.com" in host:
        qs = parse_qs(parsed.query)
        if qs.get("v"):
            return f"youtube.com/watch?v={qs['v'][0]}"
    if "bilibili.com" in host:
        return host + parsed.path
    return host + parsed.path


NORMALIZE_TARGET_DBFS = -19.0
MIN_SEGMENT_RMS_DBFS = -35.0
MIN_SEGMENT_PEAK_DBFS = -12.0
MAX_SILENT_RATIO = 0.40
MIN_SPEECH_RATIO = 0.60
MIN_SEGMENT_DURATION_SEC = 2.0
MAX_SEGMENT_DURATION_SEC = 12.0
SILENT_FRAME_DBFS = -40.0
SILENT_FRAME_MS = 30


def normalize_to_dbfs(audio, target_dbfs: float = NORMALIZE_TARGET_DBFS):
    if len(audio) == 0:
        return audio
    current = float(audio.dBFS)
    if current == float("-inf") or current != current:
        return audio
    return audio.apply_gain(target_dbfs - current)


def path_key(p: str | Path) -> str:
    return os.path.normcase(Path(p).resolve().as_posix())


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
        return False, (proc.stderr or "").strip()[-1000:]
    return True, ""


def segment_audio_metrics(audio) -> dict[str, float]:
    import numpy as np

    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32)
    if samples.size == 0:
        return {
            "rms_dbfs": -999.0,
            "peak_dbfs": -999.0,
            "silent_ratio": 1.0,
            "speech_ratio": 0.0,
            "duration_sec": 0.0,
        }
    if getattr(audio, "channels", 1) > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    max_possible = float(getattr(audio, "max_possible_amplitude", 32768.0) or 32768.0)
    samples = samples / max_possible
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    rms_dbfs = 20.0 * float(np.log10(max(rms, 1e-12)))
    peak_dbfs = 20.0 * float(np.log10(max(peak, 1e-12)))

    frame_len = max(1, int(audio.frame_rate * SILENT_FRAME_MS / 1000))
    frame_count = len(samples) // frame_len
    if frame_count <= 0:
        silent_ratio = 1.0
    else:
        frames = samples[: frame_count * frame_len].reshape(frame_count, frame_len)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        frame_dbfs = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
        silent_ratio = float(np.mean(frame_dbfs <= SILENT_FRAME_DBFS))

    duration_sec = float(len(samples)) / float(getattr(audio, "frame_rate", 16000) or 16000)
    return {
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "silent_ratio": silent_ratio,
        "speech_ratio": max(0.0, 1.0 - silent_ratio),
        "duration_sec": duration_sec,
    }


def segment_passes_quality(metrics: dict[str, float]) -> bool:
    duration_sec = float(metrics.get("duration_sec", 0.0))
    rms_dbfs = float(metrics.get("rms_dbfs", -999.0))
    peak_dbfs = float(metrics.get("peak_dbfs", -999.0))
    silent_ratio = float(metrics.get("silent_ratio", 1.0))
    speech_ratio = float(metrics.get("speech_ratio", 0.0))
    if duration_sec < MIN_SEGMENT_DURATION_SEC or duration_sec > MAX_SEGMENT_DURATION_SEC:
        return False
    if rms_dbfs <= MIN_SEGMENT_RMS_DBFS:
        return False
    if peak_dbfs <= MIN_SEGMENT_PEAK_DBFS:
        return False
    if silent_ratio >= MAX_SILENT_RATIO:
        return False
    if speech_ratio < MIN_SPEECH_RATIO:
        return False
    return True


def move_segment_to_bucket(seg_path: Path, bucket_dir: Path, out_name: str) -> Path:
    bucket_dir.mkdir(parents=True, exist_ok=True)
    out_path = bucket_dir / out_name
    if out_path.exists():
        out_path.unlink()
    seg_path.replace(out_path)
    return out_path


def download_audio(url: str, out_dir: Path, headers: dict | None = None, proxy: str | None = None) -> Path | None:
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }
    if headers:
        ydl_opts["http_headers"] = headers
    if proxy:
        ydl_opts["proxy"] = proxy
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            guessed = Path(ydl.prepare_filename(info)).with_suffix(".wav")
            if guessed.exists():
                return guessed
            wavs = sorted(out_dir.glob("*.wav"), key=os.path.getmtime)
            return wavs[-1] if wavs else None
    except Exception as exc:
        print(f"  download failed: {exc}")
        return None


def compute_mfcc_embedding(wav_path: Path):
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav_path), sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_d = librosa.feature.delta(mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc_d.mean(axis=1)])
    return feat / (np.linalg.norm(feat) + 1e-10)


def parse_json_from_text(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty output")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def load_reference_embedding(
    pip,
    speaker_ref: Path,
    example_ref: Path | None,
    converted_dir: Path,
    mfcc_thres: float,
) -> tuple[bool, Any, Any, Path | None]:
    helper = Path(__file__).resolve().parent / "encoder_cli.py"
    ref_conv: Path | None = None
    mfcc_ref_emb = compute_mfcc_embedding(example_ref if example_ref and example_ref.exists() else speaker_ref)
    if example_ref and example_ref.exists():
        ref_conv = converted_dir / "example_ref_16k.wav"
        pip.convert_audio(example_ref, ref_conv)
    elif speaker_ref.exists():
        ref_conv = converted_dir / "ref_16k.wav"
        pip.convert_audio(speaker_ref, ref_conv)
    if ref_conv and ref_conv.exists():
        ref_pcm = converted_dir / f"{ref_conv.stem}_pcm16.wav"
        ok, _ = ensure_pcm16_wav(ref_conv, ref_pcm)
        if ok:
            ref_conv = ref_pcm

    use_encoder = False
    encoder_ref_emb = None
    if helper.exists() and ref_conv and ref_conv.exists():
        try:
            cmd = [
                sys.executable,
                str(helper),
                "--model",
                Path(pip.CFG.get("mb_encoder")).resolve().as_posix(),
                "--ref",
                Path(ref_conv).resolve().as_posix(),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                out_text = proc.stdout.strip()
                data = parse_json_from_text(out_text)
                if data.get("status") == "ok" and "emb" in data:
                    encoder_ref_emb = data["emb"]
                    use_encoder = True
        except Exception:
            pass

    return use_encoder, encoder_ref_emb, mfcc_ref_emb, ref_conv


def route_segment(seg_path: Path, out_name: str, status: str, pass_dir: Path, uncertain_dir: Path) -> Path:
    bucket = pass_dir if status == "pass" else uncertain_dir
    return move_segment_to_bucket(seg_path, bucket, out_name)


def main():
    parser = ArgumentParser(description="Verify and download candidates")
    parser.add_argument("--candidates", type=str, default="../../pipeline_temp/download_candidates.json")
    parser.add_argument("--speaker-ref", type=str, required=True, help="speaker reference audio path or directory")
    parser.add_argument("--example-ref", type=str, default=None, help="example reference segment path")
    parser.add_argument("--limit", type=int, default=5, help="max number of candidates to process")
    parser.add_argument("--out", type=str, default="../../pipeline_temp/review", help="review output root")
    parser.add_argument("--min-segs", type=int, default=1, help="minimum kept segments per source")
    parser.add_argument("--keep-original", action="store_true", help="keep original downloaded media")
    parser.add_argument("--mfcc-thres", type=float, default=0.6, help="fallback MFCC threshold")
    parser.add_argument("--force", action="store_true", help="force reprocess existing review_meta.json")
    args = parser.parse_args()

    pip = load_pipeline_module()
    cand_path = Path(args.candidates).resolve()
    if not cand_path.exists():
        print(f"candidate file not found: {cand_path}")
        sys.exit(1)

    speaker_ref = Path(args.speaker_ref).resolve()
    if not speaker_ref.exists():
        print(f"speaker reference not found: {speaker_ref}")
        sys.exit(1)

    data = json.loads(cand_path.read_text(encoding="utf-8"))
    seen = set()
    items: list[tuple[str, str, str]] = []
    for platform, entries in data.items():
        for entry in entries:
            url = entry.get("url") or entry.get("webpage_url") or entry.get("id")
            if not url:
                continue
            key = canonicalize_url(url)
            if key in seen:
                continue
            seen.add(key)
            items.append((platform, entry.get("title") or "", url))
            if args.limit and len(items) >= args.limit:
                break
        if args.limit and len(items) >= args.limit:
            break

    temp_root = (Path(__file__).resolve().parents[1] / "pipeline_temp").resolve()
    raw_dir = temp_root / "raw"
    converted_dir = temp_root / "converted"
    review_dir = Path(args.out).resolve()
    pass_dir = review_dir / "pass"
    uncertain_dir = review_dir / "uncertain"
    for d in (raw_dir, converted_dir, review_dir, pass_dir, uncertain_dir):
        d.mkdir(parents=True, exist_ok=True)

    def save_segment(seg_path: Path, out_name: str, status: str) -> Path:
        return route_segment(seg_path, out_name, status, pass_dir, uncertain_dir)

    print(f"Candidates after dedupe: {len(items)}")

    ref_conv = None
    use_encoder, encoder_ref_emb, mfcc_ref_emb, ref_conv = load_reference_embedding(
        pip,
        speaker_ref=speaker_ref,
        example_ref=Path(args.example_ref).resolve() if args.example_ref else None,
        converted_dir=converted_dir,
        mfcc_thres=float(args.mfcc_thres),
    )

    try:
        meta_file = review_dir / "review_meta.json"
        if meta_file.exists() and not args.force:
            existing = json.loads(meta_file.read_text(encoding="utf-8"))
            seen_urls = {canonicalize_url(entry.get("url")) for entry in existing if entry.get("url")}
            if seen_urls:
                old_count = len(items)
                items = [item for item in items if canonicalize_url(item[2]) not in seen_urls]
                print(f"Skipped {old_count - len(items)} already processed items; remaining {len(items)}")
        elif args.force:
            print("Force reprocessing all candidates")
    except Exception:
        pass

    results = []
    for idx, (platform, title, url) in enumerate(items, 1):
        print(f"\n[{idx}/{len(items)}] {platform} - {title[:80]}")
        try:
            host = pip.anti_scraping.get_platform_host(platform) or None
            pip.anti_scraping._wait_for_host(host)
        except Exception:
            pass

        headers = pip.anti_scraping.get_platform_headers(platform)
        downloaded = download_audio(url, raw_dir, headers=headers, proxy=None)
        if not downloaded:
            print("  download failed, skipped")
            continue
        print(f"  downloaded: {downloaded.name}")

        if args.keep_original:
            shutil.copy2(downloaded, review_dir / downloaded.name)

        proc_source = downloaded
        try:
            demucs_out = Path(pip.CFG.get("demucs_out") or (Path(pip.CFG.get("temp_dir")) / "demucs"))
            vocals = pip.separate_vocals(downloaded, demucs_out)
            if vocals:
                proc_source = vocals
                print(f"  demucs vocals: {vocals.name}")
        except Exception:
            pass

        conv = converted_dir / f"{proc_source.stem}_16k.wav"
        try:
            pip.convert_audio(proc_source, conv)
        except Exception as exc:
            print(f"  convert failed: {exc}")
            continue

        try:
            if not pip.CFG.get("allow_music", False) and pip.is_likely_music_audio(conv):
                print("  likely music, skipped")
                conv.unlink(missing_ok=True)
                continue
        except Exception:
            pass

        audio = pip.AudioSegment.from_wav(conv)
        segs = pip.split_audio(audio, downloaded.stem)
        seg_export_list: list[tuple[int, Path, dict[str, float]]] = []
        for sidx, seg in enumerate(segs, 1):
            seg_name = f"{downloaded.stem}_seg{sidx:03d}.wav"
            seg_path = converted_dir / seg_name
            seg = normalize_to_dbfs(seg, NORMALIZE_TARGET_DBFS)
            seg.export(seg_path, format="wav")

            seg_pcm = converted_dir / f"{seg_path.stem}_pcm16.wav"
            ok, err = ensure_pcm16_wav(seg_path, seg_pcm)
            if not ok:
                print(f"  pcm16 convert failed: {err}")
                seg_path.unlink(missing_ok=True)
                continue
            seg_path.unlink(missing_ok=True)
            seg_path = seg_pcm

            try:
                exported = pip.AudioSegment.from_wav(seg_path)
                metrics = segment_audio_metrics(exported)
                if metrics["rms_dbfs"] < MIN_SEGMENT_RMS_DBFS and metrics["silent_ratio"] < 0.80:
                    boosted = normalize_to_dbfs(exported, -19.0)
                    boosted.export(seg_path, format="wav")
                    exported = pip.AudioSegment.from_wav(seg_path)
                    metrics = segment_audio_metrics(exported)
                if not segment_passes_quality(metrics):
                    seg_path.unlink(missing_ok=True)
                    continue
            except Exception:
                seg_path.unlink(missing_ok=True)
                continue

            try:
                if not pip.is_clean_speech(seg_path):
                    seg_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass

            seg_export_list.append((sidx, seg_path, metrics))

        kept = 0
        meta: list[dict[str, Any]] = []
        if use_encoder and seg_export_list:
            try:
                targets = [Path(p).resolve().as_posix() for _, p, _ in seg_export_list]
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "encoder_cli.py"),
                    "--model",
                    Path(pip.CFG.get("mb_encoder")).resolve().as_posix(),
                    "--ref",
                    Path(ref_conv).resolve().as_posix(),
                    "--targets",
                ] + targets
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                sims_map: dict[str, dict[str, Any]] = {}
                if proc.returncode == 0:
                    try:
                        out_text = proc.stdout.strip()
                        parsed = parse_json_from_text(out_text)
                        if parsed.get("status") == "ok" and "sims" in parsed:
                            sims_map = {
                                path_key(item.get("target_key") or item.get("target")): item
                                for item in parsed["sims"]
                            }
                    except Exception:
                        sims_map = {}
                for sidx, seg_path, metrics in seg_export_list:
                    vp = sims_map.get(path_key(seg_path))
                    sim = float(vp.get("sim", 0.0)) if vp else 0.0
                    out_name = f"{platform}_{idx:03d}_seg{sidx:03d}.wav"
                    if sim >= 0.82:
                        save_segment(seg_path, out_name, "pass")
                        status = "pass"
                    elif sim >= 0.70:
                        save_segment(seg_path, out_name, "uncertain")
                        status = "uncertain"
                    else:
                        seg_path.unlink(missing_ok=True)
                        status = "reject"
                    if status != "reject":
                        kept += 1
                        meta.append({"segment": out_name, "sim": sim, "status": status, **metrics})
            except Exception:
                for sidx, seg_path, metrics in seg_export_list:
                    try:
                        feat = compute_mfcc_embedding(seg_path)
                        sim = float((feat @ mfcc_ref_emb).astype(float))
                    except Exception:
                        sim = 0.0
                    out_name = f"{platform}_{idx:03d}_seg{sidx:03d}.wav"
                    if sim >= 0.75:
                        save_segment(seg_path, out_name, "uncertain")
                        kept += 1
                        meta.append({"segment": out_name, "sim": sim, "status": "uncertain", **metrics})
                    else:
                        seg_path.unlink(missing_ok=True)
        else:
            for sidx, seg_path, metrics in seg_export_list:
                try:
                    feat = compute_mfcc_embedding(seg_path)
                    sim = float((feat @ mfcc_ref_emb).astype(float))
                except Exception:
                    sim = 0.0
                out_name = f"{platform}_{idx:03d}_seg{sidx:03d}.wav"
                if sim >= 0.75:
                    save_segment(seg_path, out_name, "uncertain")
                    kept += 1
                    meta.append({"segment": out_name, "sim": sim, "status": "uncertain", **metrics})
                else:
                    seg_path.unlink(missing_ok=True)

        results.append({"platform": platform, "title": title, "url": url, "kept_segments": meta})
        print(f"  kept segments: {len(meta)}")

    meta_path = review_dir / "review_meta.json"
    meta_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Review directory: {review_dir}")


if __name__ == "__main__":
    main()
