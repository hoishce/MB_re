#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化语音数据流水线
══════════════════════════════════════════════════════════════
流程：爬虫下载音频 → 预处理(16kHz/单声道/WAV) → 自动切片
      → faster-whisper 转录 → 整理进 Mockingbird 训练目录
══════════════════════════════════════════════════════════════

输出目录结构(Mockingbird 可直接读取）：

  mockingbird_dataset/
  ├── wavs/                  ← 16kHz · 单声道 · WAV 切片
  │   ├── seg_0001.wav
  │   └── ...
  ├── transcripts/           ← 每条切片对应的 .txt(含时间戳)
  │   ├── seg_0001.txt
  │   └── ...
  └── pipeline_log.json      ← 本次运行日志

用法：
  python pipeline.py                        # 交互模式
  python pipeline.py --url <URL>            # 直接下载单个链接
  python pipeline.py --keyword "XXX" --n 5  # 关键词搜索下载
"""

# ── 修复 OpenMP 库冲突（与你的 whisper.py 一致）──────────────────────────────
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── 强制加载 CUDA DLL（与你的 whisper.py 一致）───────────────────────────────
_CUDA_DLL_DIR = os.environ.get("MOCKINGBIRD_CUDA_DLL_DIR")
if _CUDA_DLL_DIR and os.path.exists(_CUDA_DLL_DIR):
    os.add_dll_directory(_CUDA_DLL_DIR)

# 确保仓库根能被 import 查找到，再导入并调用 ensure_dlls
from pathlib import Path as _Path
try:
    _p = _Path(__file__).resolve()
    if str(_p.parent) not in sys.path:
        sys.path.insert(0, str(_p.parent))
    for _parent in _p.parents:
        if (_parent / "models").is_dir():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break
except Exception:
    pass
try:
    import ensure_dlls
    ensure_dlls.ensure_dll_priority()
except Exception:
    pass

import re, json, time, shutil, argparse, subprocess
import wave
from pathlib import Path
from datetime import datetime
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    _p = Path(__file__).resolve()
    for _parent in _p.parents:
        if (_parent / "models").is_dir():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break
except Exception:
    pass

# ── 自动安装缺失依赖 ──────────────────────────────────────────────────────────
def _ensure(pkg, import_as=None):
    try:
        __import__(import_as or pkg)
    except ImportError:
        print(f"  [安装] {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("yt_dlp",          "yt_dlp")
_ensure("rich")
_ensure("faster-whisper",  "faster_whisper")
_ensure("pydub")
_ensure("soundfile")
_ensure("requests")
_ensure("librosa")

import torch
import numpy as np
import librosa
import yt_dlp
from faster_whisper import WhisperModel
from pydub import AudioSegment, silence as pydub_silence
from rich.console  import Console
from rich.panel    import Panel
from rich.table    import Table
from rich.rule     import Rule
from rich.prompt   import Prompt, Confirm
from rich.progress import (Progress, SpinnerColumn, BarColumn,
                            TextColumn, TimeElapsedColumn, TaskProgressColumn)
import anti_scraping


def _require_compatible_numpy() -> None:
    """Fail fast when the environment has an incompatible NumPy release."""
    version = np.__version__.split("+", 1)[0]
    profile = os.environ.get("MOCKINGBIRD_PIPELINE_PROFILE", "core").strip().lower() or "core"
    if profile == "advanced":
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        if (major, minor, patch) < (2, 2, 2):
            raise RuntimeError(
                f"Detected NumPy {np.__version__}, but the advanced profile expects NumPy 2.2.2 or newer. "
                "Please run the advanced stages inside the advanced environment."
            )
    else:
        if version != "1.26.4":
            raise RuntimeError(
                f"Detected NumPy {np.__version__}, but this project expects NumPy 1.26.4. "
                "Please reinstall dependencies from requirements.txt and keep the pinned "
                "version (numpy==1.26.4)."
            )


_require_compatible_numpy()

# Optional imports that may be unavailable on some platforms (Windows).
# Provide stub variables so static analyzers won't report unresolved imports.
try:
    import webrtcvad as _webrtcvad  # type: ignore
except Exception:
    _webrtcvad = None

console = Console()

# ════════════════════════════════════════════════════════════════════════════
#  ★  配置区  ← 按你的实际路径修改这里
# ════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Whisper（复用你的 whisper.py 设置）─────────────────────────────────
    "whisper_model_dir" : os.environ.get("MOCKINGBIRD_WHISPER_MODEL", "large-v3"),
    "whisper_device"    : "cuda",
    "whisper_index"     : 0,
    "whisper_compute"   : "float16",   # 4060 推荐 float16
    "whisper_beam"      : 1,           # beam_size=1 速度最快
    "whisper_vad"       : True,        # VAD 过滤静音
    "whisper_lang"      : None,        # None=自动检测, "zh"=强制中文

    # ── Mockingbird 模型路径（复用你的 mockingbird-01.py 设置）────────────
    "mb_encoder"    : os.environ.get("MOCKINGBIRD_ENCODER_CKPT", str(REPO_ROOT / "saved_models" / "encoder" / "saved_models" / "pretrained.pt")),
    "mb_synthesizer": os.environ.get("MOCKINGBIRD_SYNTHESIZER_CKPT", str(REPO_ROOT / "saved_models" / "synthesizer" / "saved_models" / "mandarin")),
    "mb_vocoder"    : os.environ.get("MOCKINGBIRD_VOCODER_CKPT", str(REPO_ROOT / "saved_models" / "vocoder" / "saved_models" / "pretrained" / "pretrained.pt")),

    # ── 输出目录 ────────────────────────────────────────────────────────────
    "output_root"   : os.environ.get("MOCKINGBIRD_OUTPUT_ROOT", str(REPO_ROOT / "mockingbird_dataset")),
    "temp_dir"      : os.environ.get("MOCKINGBIRD_TEMP_DIR", str(REPO_ROOT / "pipeline_temp")),

    # ── 爬虫 ────────────────────────────────────────────────────────────────
    "platform"      : "ytsearch",
    "multi_platform": [],
    "max_results"   : 10,

    # ── 音频预处理 ──────────────────────────────────────────────────────────
    "target_sr"         : 16000,
    "target_channels"   : 1,
    "min_seg_sec"       : 2.0,
    "max_seg_sec"       : 15.0,
    "silence_thresh_db" : -40,
    "min_silence_ms"    : 300,

    # ── 过滤设置 ────────────────────────────────────────────────────────────
    "allow_music"                   : False,
    "use_speaker_verification"      : False,
    "speaker_ref"                   : None,
    "speaker_similarity_threshold"  : 0.85,
    "filter_multi_speaker"          : True,
    "multi_speaker_similarity"      : 0.90,
    # Demucs / 音源分离配置
    "use_demucs"                    : True,
    "demucs_cpu"                    : True,
    "demucs_out"                    : None,   # 默认使用 temp_dir/demucs
    # 音乐检测（尝试 audioset/panns，否则回退到启发式检测）
    "use_audioset_classifier"       : False,
    "music_detection_threshold"     : 0.5,
    # 更严格的候选级相似度（平均分）以排除假阳性
    "candidate_similarity_threshold" : 0.78,
    # 聚类相关阈值，用于同一来源内分片聚类
    "cluster_merge_threshold"       : 0.80,
    "cluster_min_members"           : 1,
}

if torch.cuda.is_available():
    CFG["whisper_device"] = "cuda"
    CFG["whisper_compute"] = "float16"
else:
    CFG["whisper_device"] = "cpu"
    CFG["whisper_compute"] = "float32"

if os.environ.get("MOCKINGBIRD_PIPELINE_PROFILE", "core").strip().lower() == "advanced":
    CFG["whisper_beam"] = 5
    CFG["whisper_vad"] = False
    CFG["whisper_lang"] = "zh"
# ════════════════════════════════════════════════════════════════════════════


# ─── 工具 ────────────────────────────────────────────────────────────────────
def banner():
    console.print(Panel.fit(
        "[bold cyan]🔄 语音数据自动化流水线[/bold cyan]\n"
        "[dim]爬虫  ▸  预处理(16kHz/单声道/WAV)  ▸  切片  ▸  Whisper转录  ▸  Mockingbird数据集[/dim]",
        border_style="cyan", padding=(1, 4),
    ))
    console.print()

def safe_name(s: str, maxlen=50) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", s)[:maxlen]

def ms_to_hms(ms: float) -> str:
    s = int(ms / 1000)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def _platform_name(code: str) -> str:
    names = {
        "ytsearch": "YouTube", "bilisearch": "Bilibili",
        "scsearch": "SoundCloud", "niconico": "Niconico",
        "odysee": "Odysee", "peertube": "PeerTube",
        "mixcloud": "Mixcloud", "archiveorg": "Archive.org",
        "cda": "CDA.pl", "rutube": "Rutube",
    }
    return names.get(code, code)

def gpu_status():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green][GPU] {name}  显存：{vram:.1f} GB[/green]")
    else:
        console.print("[yellow][WARN] 未检测到 CUDA，将使用 CPU（速度较慢）[/yellow]")
    console.print()


MUSIC_KEYWORDS = [
    "mv", "official", "lyrics", "翻唱", "cover", "歌曲", "music", "audio",
    "ost", "live", "remix", "single", "ft.", "feat.", "演唱", "伴奏"
]

def is_probable_music(entry: dict) -> bool:
    title    = (entry.get("title")    or "").lower()
    uploader = (entry.get("uploader") or "").lower()
    desc     = (entry.get("description") or "").lower()
    for k in MUSIC_KEYWORDS:
        if k in title or k in uploader or k in desc:
            return True
    return False


def separate_vocals(src: Path, out_dir: Path) -> Path | None:
    """使用 demucs 将 `src` 分离人声(vocals),返回 vocals 文件路径或 None。
    优先使用 demucs CLI(避免在主进程里导入大量依赖),当 CLI 不可用时回退为 None。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        env = os.environ.copy()
        if CFG.get('demucs_cpu', True):
            env['CUDA_VISIBLE_DEVICES'] = ''
        cmd = ['demucs', '--two-stems=vocals', '--out', str(out_dir), str(src)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if proc.returncode != 0:
            return None
    except Exception:
        return None

    # demucs 常见输出路径：<out_dir>/<model>/<filename>/vocals.wav 或 <out_dir>/<filename>/vocals.wav
    candidates = list(out_dir.rglob('*vocals*.wav'))
    if not candidates:
        candidates = list(out_dir.rglob('vocals.wav'))
    if not candidates:
        # 尝试按源文件名匹配
        candidates = list(out_dir.rglob(f"{src.stem}*vocals*.wav"))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def is_likely_music_audio(wav_path: Path) -> bool:
    """尝试用轻量方法判断音频是否为音乐（如果安装了 audioset/panns 会优先使用）。
    回退到启发式特征(HPSS、谱平坦度、过零率)。
    """
    # 为了避免编辑器/静态分析器在未安装 panns_inference 时报错，
    # 预先定义符号并在导入失败时回退为 None。
    SoundEventModel = None
    try:
        # 若安装了 panns_inference 可在此处接入更强的音频标签器（可选）
        from panns_inference import SoundEventModel  # type: ignore
        # 若存在更健全的库，用户可以启用 CFG['use_audioset_classifier']
    except Exception:
        SoundEventModel = None

    try:
        y, sr = librosa.load(str(wav_path), sr=CFG.get('target_sr', 16000))
        harmonic, percussive = librosa.effects.hpss(y)
        hr = float(np.sum(np.abs(harmonic)) / (np.sum(np.abs(y)) + 1e-10))
        flatness = float(librosa.feature.spectral_flatness(y=y).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        # 启发式判定：高谐波比 + 低谱平坦度 + 低过零率 更可能是乐音
        if hr > 0.45 and flatness < 0.35 and zcr < 0.12:
            return True
    except Exception:
        return False
    return False


def vad_speech_ratio(wav_path: Path, aggressiveness: int = 2) -> float:
    """使用 webrtcvad 计算音频中被判定为语音的帧比例；若不可用回退到 librosa 能量分段比率。"""
    try:
        # 使用模块顶部的可选导入 `_webrtcvad`，避免在函数内再次导入造成编辑器解析问题
        webrtcvad_local = _webrtcvad
        import wave as _wave
        with _wave.open(str(wav_path), 'rb') as wf:
            sr = wf.getframerate()
            width = wf.getsampwidth()
            nch = wf.getnchannels()
            if sr != CFG.get('target_sr', 16000) or width != 2 or nch != 1:
                # 重新采样并导出 PCM bytes
                y, _ = librosa.load(str(wav_path), sr=CFG.get('target_sr', 16000))
                pcm = (y * 32767).astype(np.int16).tobytes()
                sample_rate = CFG.get('target_sr', 16000)
            else:
                pcm = wf.readframes(wf.getnframes())
                sample_rate = sr
        if webrtcvad_local is None:
            raise ImportError('webrtcvad not available')

        vad = webrtcvad_local.Vad(int(aggressiveness))
        frame_ms = 30
        frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        frames = [pcm[i:i+frame_bytes] for i in range(0, len(pcm), frame_bytes) if len(pcm[i:i+frame_bytes])==frame_bytes]
        if not frames:
            return 0.0
        speech_frames = sum(1 for f in frames if vad.is_speech(f, sample_rate))
        return float(speech_frames) / float(len(frames))
    except Exception:
        # 回退：librosa 能量检测
        try:
            y, sr = librosa.load(str(wav_path), sr=None)
            intervals = librosa.effects.split(y, top_db=30)
            speech_ratio = sum(e - s for s, e in intervals) / len(y)
            return float(speech_ratio)
        except Exception:
            return 0.0


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — 爬虫下载
# ════════════════════════════════════════════════════════════════════════════
def step_download(urls: list[str]) -> list[Path]:
    temp = Path(CFG["temp_dir"]) / "raw"
    temp.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format"        : "bestaudio/best",
        "outtmpl"       : str(temp / "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key"             : "FFmpegExtractAudio",
            "preferredcodec"  : "wav",
            "preferredquality": "0",
        }],
        "quiet"      : True,
        "no_warnings": True,
    }

    downloaded: list[Path] = []
    console.print(Rule("[bold]STEP 1  爬虫下载[/bold]"))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("下载中...", total=len(urls))
        for url in urls:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info  = ydl.extract_info(url, download=True)
                    fpath = Path(ydl.prepare_filename(info)).with_suffix(".wav")
                    if not fpath.exists():
                        wavs  = sorted(temp.glob("*.wav"), key=os.path.getmtime)
                        fpath = wavs[-1] if wavs else None
                    if fpath and fpath.exists():
                        downloaded.append(fpath)
                        prog.print(f"  [green][OK][/green] {fpath.name}")
                    else:
                        prog.print(f"  [red][ERR] 找不到输出文件：{url}[/red]")
            except Exception as e:
                prog.print(f"  [red][ERR] 下载失败：{e}[/red]")
            prog.advance(task)

    console.print(f"[bold green]  下载完成：{len(downloaded)} 个文件[/bold green]\n")
    return downloaded


def search_urls(keyword: str, n: int, platform: str | None = None) -> list[str]:
    platforms = CFG.get("multi_platform", []) if not platform else [platform]
    if not platforms:
        platforms = [CFG.get("platform", "ytsearch")]

    all_entries = []
    console.print(f"  搜索：[cyan]{keyword}[/cyan] ({', '.join([_platform_name(p) for p in platforms])})")

    for plat in platforms:
        try:
            if plat == "archiveorg":
                import urllib.request as _ur
                import urllib.parse as _up
                import json as _json
                params = {
                    'q': f'title:("{keyword}") OR creator:("{keyword}")',
                    'fl[]': ['identifier', 'title', 'creator', 'mediatype'],
                    'rows': n, 'output': 'json',
                }
                url = 'https://archive.org/advancedsearch.php?' + _up.urlencode(params, doseq=True)
                with _ur.urlopen(url, timeout=10) as resp:
                    data = _json.load(resp)
                for d in data.get("response", {}).get("docs", []):
                    identifier = d.get("identifier")
                    all_entries.append({
                        "title": d.get("title") or identifier,
                        "uploader": d.get("creator") or "",
                        "duration": None,
                        "url": f"https://archive.org/details/{identifier}",
                        "_platform": "archiveorg",
                    })
                continue

            if plat == "mixcloud":
                try:
                    from urllib.parse import quote_plus as _qp
                    url  = f"https://api.mixcloud.com/search/?q={_qp(keyword)}&type=cloudcast&limit={n}"
                    data = anti_scraping.fetch_json_with_retry(url)
                    if data and isinstance(data, dict):
                        for d in data.get('data', []):
                            all_entries.append({
                                'title': d.get('name'), 'uploader': d.get('username'),
                                'duration': None, 'url': d.get('url'), '_platform': 'mixcloud',
                            })
                        continue
                except Exception:
                    pass

            headers = anti_scraping.get_platform_headers(plat)
            entries = anti_scraping.yt_dlp_search_for_platform(plat, n, keyword, http_headers=headers)
            for e in entries:
                e["_platform"] = plat
            all_entries.extend(entries)

        except Exception as e:
            console.print(f"[red]搜索平台 {plat} 失败：{e}[/red]")

    entries = all_entries
    if not entries:
        console.print("[yellow]所有平台均无结果[/yellow]")
        return []

    if not CFG.get("allow_music", False):
        orig    = len(entries)
        entries = [e for e in entries if not is_probable_music(e)]
        removed = orig - len(entries)
        if removed:
            console.print(f"[dim]已过滤 {removed} 个疑似音乐结果（使用 --allow-music 可禁用）[/dim]")
        if not entries:
            console.print("[yellow]注意：过滤音乐后无结果，回退显示全部。[/yellow]")
            entries = all_entries

    tbl = Table(show_lines=True, header_style="bold magenta", border_style="blue")
    tbl.add_column("序号", width=5, justify="center")
    tbl.add_column("标题", min_width=30, max_width=50)
    tbl.add_column("时长", width=9, justify="center")
    tbl.add_column("上传者", width=16)
    tbl.add_column("平台", width=10, justify="center")
    for i, e in enumerate(entries, 1):
        dur   = e.get("duration")
        dur_s = f"{int(dur)//60}:{int(dur)%60:02d}" if dur else "—"
        tbl.add_row(str(i), (e.get("title") or "")[:50],
                    dur_s, (e.get("uploader") or "")[:16], e.get("_platform", "?"))
    console.print(tbl)

    console.print("[dim]输入序号(逗号分隔),0=全部,q=跳过[/dim]")
    sel = Prompt.ask("选择").strip().lower()
    if sel == "q":   return []
    if sel == "0":   chosen = entries
    else:
        idx    = [int(x)-1 for x in sel.split(",") if x.strip().isdigit()]
        chosen = [entries[i] for i in idx if 0 <= i < len(entries)]
    return [e.get("url") or e.get("webpage_url") or e.get("id") for e in chosen]


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — 音频预处理 + 切片
# ════════════════════════════════════════════════════════════════════════════
def convert_audio(src: Path, dst: Path):
    audio = AudioSegment.from_file(src)
    audio = audio.set_frame_rate(CFG["target_sr"])
    audio = audio.set_channels(CFG["target_channels"])
    audio = audio.set_sample_width(2)
    audio.export(dst, format="wav")


def split_audio(audio: AudioSegment, src_name: str) -> list[AudioSegment]:
    min_ms = int(CFG["min_seg_sec"] * 1000)
    max_ms = int(CFG["max_seg_sec"] * 1000)
    chunks = pydub_silence.split_on_silence(
        audio,
        min_silence_len = CFG["min_silence_ms"],
        silence_thresh  = CFG["silence_thresh_db"],
        keep_silence    = 200,
    )
    if not chunks:
        chunks = [audio]
    result = []
    for chunk in chunks:
        duration = len(chunk)
        if duration < min_ms:
            continue
        if duration <= max_ms:
            result.append(chunk)
        else:
            for start in range(0, duration, max_ms):
                sub = chunk[start: start + max_ms]
                if len(sub) >= min_ms:
                    result.append(sub)
    return result


def is_clean_speech(wav_path: Path) -> bool:
    # 优先使用 webrtcvad（如可用），否则回退到 librosa 能量分段法
    try:
        sr = CFG.get('target_sr', 16000)
        speech_ratio = vad_speech_ratio(wav_path, aggressiveness=CFG.get('webrtcvad_aggressiveness', 2))
        if speech_ratio < 0.4:
            return False
    except Exception:
        try:
            y, sr = librosa.load(str(wav_path), sr=None)
            intervals = librosa.effects.split(y, top_db=30)
            speech_ratio = sum(e - s for s, e in intervals) / len(y)
            if speech_ratio < 0.6:
                return False
        except Exception:
            return False

    # 简单的谱平坦度检测（乐音通常谱平坦度较低）
    try:
        y, sr = librosa.load(str(wav_path), sr=CFG.get('target_sr', 16000))
        flatness = float(librosa.feature.spectral_flatness(y=y).mean())
        if flatness > 0.35:
            return False
    except Exception:
        pass

    # 若启用了 audioset/panns 分类器或启发式音乐检测，排除明显音乐
    try:
        if not CFG.get('allow_music', False) and is_likely_music_audio(wav_path):
            return False
    except Exception:
        pass

    return True


def ensure_encoder_loaded():
    try:
        from models.encoder import inference as encoder
    except Exception as e:
        console.print(f"[yellow]无法导入 encoder: {e}[/yellow]")
        return None
    try:
        if not encoder.is_loaded():
            try:
                encoder.load_model(Path(CFG["mb_encoder"]))
            except Exception as e_gpu:
                console.print(f"[yellow]GPU 加载失败: {e_gpu}，尝试 CPU...[/yellow]")
                try:
                    encoder.load_model(Path(CFG["mb_encoder"]), device="cpu")
                except Exception as e_cpu:
                    console.print(f"[yellow]CPU 加载也失败: {e_cpu}[/yellow]")
                    return None
    except Exception as e:
        console.print(f"[yellow]加载 encoder 模型失败: {e}[/yellow]")
        return None
    return encoder


def compute_embedding_for_wav(encoder, wav_path: Path):
    wav = encoder.preprocess_wav(str(wav_path))
    return encoder.embed_utterance(wav)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def is_target_speaker_wav(encoder, wav_path: Path, ref_emb: np.ndarray, thres: float):
    try:
        emb = compute_embedding_for_wav(encoder, wav_path)
        sim = _cosine_sim(emb, ref_emb)
        return sim, sim >= thres
    except Exception as e:
        console.print(f"[yellow]声纹比对失败 {wav_path.name}: {e}[/yellow]")
        return 0.0, False


def is_single_speaker_wav(encoder, wav_path: Path, sim_threshold: float):
    try:
        wav = encoder.preprocess_wav(str(wav_path))
        embed, partials, _ = encoder.embed_utterance(wav, return_partials=True)
        if partials is None or len(partials) < 2:
            return 1.0, True
        mean_emb = np.mean(partials, axis=0)
        sims     = [_cosine_sim(p, mean_emb) for p in partials]
        avg_sim  = float(np.mean(sims))
        return avg_sim, avg_sim >= sim_threshold
    except Exception as e:
        console.print(f"[yellow]多说话人检测失败 {wav_path.name}: {e}[/yellow]")
        return 0.0, True


def _next_seg_index(wavs_dir: Path) -> int:
    existing = list(wavs_dir.glob("seg_*.wav"))
    if not existing:
        return 1
    nums = [int(re.search(r"seg_(\d+)", p.stem).group(1))
            for p in existing if re.search(r"seg_(\d+)", p.stem)]
    return max(nums) + 1 if nums else 1


def step_preprocess(raw_files: list[Path], encoder_obj=None, ref_emb=None) -> list[Path]:
    wavs_dir = Path(CFG["output_root"]) / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    seg_paths: list[Path] = []
    seg_idx = _next_seg_index(wavs_dir)

    console.print(Rule("[bold]STEP 2  音频预处理 & 切片[/bold]"))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TaskProgressColumn(),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("处理中...", total=len(raw_files))

        for raw in raw_files:
            prog.update(task, description=f"处理 {raw.name[:30]}")
            try:
                # 如果启用了 demucs，则先分离人声后再切片（提高在有背景音乐时的鲁棒性）
                proc_src = raw
                if CFG.get('use_demucs', False):
                    demucs_out = Path(CFG.get('demucs_out') or Path(CFG['temp_dir']) / 'demucs')
                    try:
                        vocals = separate_vocals(raw, demucs_out)
                        if vocals:
                            proc_src = vocals
                            prog.print(f"  使用 demucs 分离人声: {vocals.name}")
                    except Exception:
                        pass

                converted = Path(CFG["temp_dir"]) / "converted" / (proc_src.stem + "_16k.wav")
                converted.parent.mkdir(parents=True, exist_ok=True)
                convert_audio(proc_src, converted)
                audio    = AudioSegment.from_wav(converted)
                segments = split_audio(audio, raw.stem)
                kept     = 0

                # 先导出并做轻量过滤，后续再做更强的聚类/相似度判断
                seg_exported: list[Path] = []
                for sidx, seg in enumerate(segments, 1):
                    tmp_name = f"{raw.stem}_seg{sidx:03d}.wav"
                    tmp_path = Path(CFG['temp_dir']) / 'converted' / tmp_name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    seg.export(tmp_path, format='wav')

                    try:
                        if not is_clean_speech(tmp_path):
                            prog.print(f"  [yellow]过滤: 非人声/音乐 → {tmp_path.name}[/yellow]")
                            tmp_path.unlink(missing_ok=True)
                            continue
                    except Exception as e:
                        prog.print(f"  [yellow]检测失败 {tmp_path.name}: {e}[/yellow]")
                        tmp_path.unlink(missing_ok=True)
                        continue

                    seg_exported.append(tmp_path)

                # 若可用 encoder 与参考 embedding，逐片段按相似度阈值判断（不使用平均相似度）
                if encoder_obj is not None and ref_emb is not None and seg_exported:
                    for p in seg_exported:
                        try:
                            emb = compute_embedding_for_wav(encoder_obj, p)
                        except Exception:
                            emb = None
                        if emb is None:
                            p.unlink(missing_ok=True)
                            continue
                        sim = _cosine_sim(emb, ref_emb)
                        if sim >= CFG.get('speaker_similarity_threshold', 0.85):
                            name = f"seg_{seg_idx:04d}.wav"
                            outp = wavs_dir / name
                            p.replace(outp)
                            seg_paths.append(outp)
                            seg_idx += 1
                            kept += 1
                        else:
                            p.unlink(missing_ok=True)

                else:
                    # 无 encoder 或参考 embedding，按单片直接判断并保存
                    for tmp_path in seg_exported:
                        out_name = f"seg_{seg_idx:04d}.wav"
                        out = wavs_dir / out_name
                        # 若需要多说话人检测或参考比对，可在此处调用现有函数
                        if encoder_obj is not None and CFG.get('filter_multi_speaker', True):
                            try:
                                avg_sim, single = is_single_speaker_wav(encoder_obj, tmp_path, CFG.get('multi_speaker_similarity', 0.90))
                                if not single:
                                    tmp_path.unlink(missing_ok=True)
                                    continue
                            except Exception:
                                pass

                        if ref_emb is not None and encoder_obj is not None:
                            try:
                                sim, match = is_target_speaker_wav(encoder_obj, tmp_path, ref_emb, CFG.get('speaker_similarity_threshold', 0.80))
                                if not match:
                                    tmp_path.unlink(missing_ok=True)
                                    continue
                            except Exception:
                                tmp_path.unlink(missing_ok=True)
                                continue

                        tmp_path.replace(out)
                        seg_paths.append(out)
                        seg_idx += 1
                        kept += 1

                prog.print(f"  [green][OK][/green] {raw.name} → {kept}/{len(segments)} 段")

            except Exception as e:
                prog.print(f"  [red][ERR] 预处理失败 {raw.name}:{e}[/red]")
            prog.advance(task)

    console.print(f"[bold green]  切片完成：共 {len(seg_paths)} 段[/bold green]\n")
    return seg_paths


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — faster-whisper 转录
# ════════════════════════════════════════════════════════════════════════════
_whisper_model: Optional[WhisperModel] = None

def load_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        console.print(f"  加载 Whisper 模型（{CFG['whisper_device'].upper()} · {CFG['whisper_compute']})...")
        _whisper_model = WhisperModel(
            CFG["whisper_model_dir"],
            device           = CFG["whisper_device"],
            device_index     = CFG["whisper_index"],
            compute_type     = CFG["whisper_compute"],
            local_files_only = True,
        )
        console.print("  [green]Whisper 模型加载完毕[/green]")
    return _whisper_model


def transcribe_one(wav_path: Path, model: WhisperModel) -> str:
    kwargs = dict(beam_size=CFG["whisper_beam"], vad_filter=CFG["whisper_vad"])
    if CFG["whisper_lang"]:
        kwargs["language"] = CFG["whisper_lang"]
    segments, _ = model.transcribe(str(wav_path), **kwargs)
    lines = [f"[{seg.start:>6.2f}s -> {seg.end:>6.2f}s] {seg.text.strip()}" for seg in segments]
    return "\n".join(lines)


def clear_whisper_model() -> None:
    global _whisper_model
    _whisper_model = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def step_transcribe(seg_paths: list[Path]) -> dict[str, str]:
    txt_dir = Path(CFG["output_root"]) / "transcripts"
    txt_dir.mkdir(parents=True, exist_ok=True)
    model   = load_whisper_model()
    results : dict[str, str] = {}

    console.print(Rule("[bold]STEP 3  Whisper 转录[/bold]"))
    t0 = time.time()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TaskProgressColumn(),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("转录中...", total=len(seg_paths))
        for wav in seg_paths:
            prog.update(task, description=f"转录 {wav.name}")
            try:
                text     = transcribe_one(wav, model)
                txt_path = txt_dir / (wav.stem + ".txt")
                txt_path.write_text(text, encoding="utf-8")
                results[wav.name] = text
                preview = " ".join(re.sub(r"\[.*?\]", "", l).strip() for l in text.splitlines())[:60]
                prog.print(f"  [green][OK][/green] {wav.name}  →  {preview}…")
            except Exception as e:
                prog.print(f"  [red][ERR] 转录失败 {wav.name}:{e}[/red]")
            prog.advance(task)

    console.print(f"[bold green]  转录完成：{len(results)} 条，耗时 {time.time()-t0:.1f}s[/bold green]\n")
    return results


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — 写入日志 & 汇总
# ════════════════════════════════════════════════════════════════════════════
def step_finalize(results: dict[str, str], seg_paths: list[Path]):
    out = Path(CFG["output_root"])
    console.print(Rule("[bold]STEP 4  写入日志[/bold]"))
    log = {
        "generated_at"  : datetime.now().isoformat(),
        "total_segments": len(seg_paths),
        "transcribed"   : len(results),
        "config"        : CFG,
        "files"         : [{"wav": w, "transcript_lines": len(t.splitlines())} for w, t in results.items()],
    }
    log_path = out / "pipeline_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    tbl = Table(title="流水线完成汇总", border_style="green", header_style="bold")
    tbl.add_column("项目",        style="bold")
    tbl.add_column("路径 / 数量", style="cyan")
    tbl.add_row("数据集根目录", str(out))
    tbl.add_row("WAV 切片",    str(out / "wavs")        + f"  ({len(seg_paths)} 个)")
    tbl.add_row("转录文本",    str(out / "transcripts") + f"  ({len(results)} 个 .txt)")
    tbl.add_row("运行日志",    str(log_path))
    console.print(tbl)
    console.print()
    console.print(Panel(
        "[bold green]全部完成！[/bold green]\n"
        "[dim]Mockingbird 训练时将 wavs/ 和 transcripts/ 路径填入配置即可。[/dim]",
        border_style="green",
    ))


# ════════════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════════════
def get_urls_interactive() -> list[str]:
    console.print("[bold]请选择音频来源：[/bold]")
    console.print("  [1] 关键词搜索")
    console.print("  [2] 直接输入 URL")
    console.print("  [3] 本地已有音频文件（跳过下载）")
    mode = Prompt.ask("选择", choices=["1","2","3"])

    if mode == "1":
        console.print("\n[bold]📡 选择搜索平台[/bold]")
        console.print("  [1] YouTube  [2] Bilibili  [3] SoundCloud")
        console.print("  [4] Niconico  [5] Odysee  [6] PeerTube")
        console.print("  [7] Mixcloud  [8] Archive.org  [9] CDA.pl  [10] Rutube")
        console.print("  [a] 所有平台")
        p = Prompt.ask("平台", default="1")
        if p.lower() == "a":
            CFG["multi_platform"] = ["ytsearch","bilisearch","scsearch","niconico",
                                     "odysee","peertube","mixcloud","archiveorg","cda","rutube"]
        else:
            platform_map = {
                "1":"ytsearch","2":"bilisearch","3":"scsearch","4":"niconico",
                "5":"odysee","6":"peertube","7":"mixcloud","8":"archiveorg","9":"cda","10":"rutube"
            }
            CFG["platform"]       = platform_map.get(p, CFG.get("platform"))
            CFG["multi_platform"] = []

        kw   = Prompt.ask("搜索关键词（多个用英文逗号分隔）")
        n    = int(Prompt.ask("每个关键词最多抓几条", default=str(CFG["max_results"])))
        urls = []
        for k in kw.split(","):
            urls.extend(search_urls(k.strip(), n))
        return urls

    elif mode == "2":
        raw = Prompt.ask("输入 URL(多个用英文逗号分隔)")
        return [u.strip() for u in raw.split(",") if u.strip()]

    else:
        Prompt.ask("本地音频文件夹路径")
        return []


def main():
    parser = argparse.ArgumentParser(description="语音数据自动化流水线")
    parser.add_argument("--url",             nargs="+")
    parser.add_argument("--keyword",         nargs="+")
    parser.add_argument("--n",               type=int,   default=CFG["max_results"])
    parser.add_argument("--local",           type=str)
    parser.add_argument("--no-download",     action="store_true")
    parser.add_argument("--speaker-ref",     type=str)
    parser.add_argument("--speaker-thres",   type=float, default=0.73)
    parser.add_argument("--allow-music",     action="store_true")
    parser.add_argument("--no-filter-multi", action="store_true")
    parser.add_argument("--platform",        type=str)
    parser.add_argument("--multi-platform",  type=str)
    parser.add_argument("--proxy",           type=str)
    parser.add_argument("--proxies-file",    type=str)
    parser.add_argument("--rate",            type=float)
    parser.add_argument("--rate-host",       type=str)
    parser.add_argument("--clean-v2",        action="store_true", help="Run cleaning_pipeline_v2 as subprocess")
    args = parser.parse_args()

    if getattr(args, 'proxy', None):
        anti_scraping.set_proxy_list([args.proxy])
    elif getattr(args, 'proxies_file', None):
        anti_scraping.load_proxies_from_file(args.proxies_file)
    else:
        anti_scraping.load_proxies_from_env()
    if anti_scraping._PROXY_LIST:
        console.print(f"[dim]使用代理池: {len(anti_scraping._PROXY_LIST)} 个代理[/dim]")

    env_rate = os.environ.get('SCRAPER_MIN_DELAY') or os.environ.get('SCRAPER_RATE')
    try:
        if getattr(args, 'rate', None) is not None:
            anti_scraping.set_global_rate_limit(args.rate)
        elif env_rate:
            anti_scraping.set_global_rate_limit(float(env_rate))
    except Exception:
        pass

    try:
        host_rates = getattr(args, 'rate_host', None) or os.environ.get('SCRAPER_HOST_DELAYS')
        if host_rates:
            for pair in [p.strip() for p in host_rates.split(',') if p.strip()]:
                if '=' in pair:
                    h, s = pair.split('=', 1)
                    try:
                        anti_scraping.set_host_rate_limit(h.strip(), float(s.strip()))
                    except Exception:
                        continue
    except Exception:
        pass

    if args.platform:
        CFG["platform"]       = args.platform
        CFG["multi_platform"] = []
    if args.multi_platform:
        CFG["multi_platform"] = [p.strip() for p in args.multi_platform.split(",") if p.strip()]
        CFG["platform"]       = CFG["multi_platform"][0] if CFG["multi_platform"] else "ytsearch"

    banner()
    gpu_status()

    # 如果请求使用 cleaning_pipeline_v2，则以子进程方式调用，避免在主进程中导入额外依赖
    if getattr(args, 'clean_v2', False):
        try:
            script = Path(__file__).resolve().parent / 'cleaning_pipeline_v2.py'
            cmd = [sys.executable, str(script)]
            # 将常用参数转发给 v2 脚本
            if getattr(args, 'local', None):
                cmd += ['--input-dir', str(Path(args.local).resolve())]
            else:
                cmd += ['--input-dir', str(Path(CFG.get('temp_dir')) / 'raw')]
            if getattr(args, 'speaker_ref', None):
                cmd += ['--speaker-ref', args.speaker_ref]
            # 若用户没有下载原始文件，可先提示
            console.print(Rule('[bold]调用 cleaning_pipeline_v2[/bold]'))
            console.print(f"[dim]命令: {' '.join(cmd)}[/dim]")
            subprocess.run(cmd)
        except Exception as e:
            console.print(f"[red]运行 cleaning_pipeline_v2 失败: {e}[/red]")
        return

    raw_files: list[Path] = []

    if args.no_download:
        raw_dir   = Path(CFG["temp_dir"]) / "raw"
        raw_files = list(raw_dir.glob("*.wav"))
        console.print(f"[yellow]跳过下载，使用已有文件 {len(raw_files)} 个[/yellow]\n")
    elif args.local:
        exts      = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
        raw_files = [p for p in Path(args.local).rglob("*") if p.suffix.lower() in exts]
        console.print(f"[cyan]本地文件：{len(raw_files)} 个[/cyan]\n")
    else:
        urls: list[str] = []
        if args.url:      urls = args.url
        elif args.keyword:
            for kw in args.keyword:
                urls.extend(search_urls(kw, args.n))
        else:
            urls = get_urls_interactive()
        if not urls:
            console.print("[red]未获取到任何 URL，退出。[/red]")
            return
        raw_files = step_download(urls)

    if not raw_files:
        console.print("[red]没有可处理的音频文件，退出。[/red]")
        return

    if getattr(args, "allow_music",     False): CFG["allow_music"]          = True
    if getattr(args, "no_filter_multi", False): CFG["filter_multi_speaker"] = False
    CFG["speaker_similarity_threshold"] = getattr(args, "speaker_thres", 0.73)

    encoder_obj = None
    ref_emb     = None
    if getattr(args, "speaker_ref", None):
        encoder_obj = ensure_encoder_loaded()
        if encoder_obj:
            try:
                ref_emb = compute_embedding_for_wav(encoder_obj, Path(args.speaker_ref))
                CFG["use_speaker_verification"] = True
                console.print(f"[green]已加载参考声纹：{args.speaker_ref}[/green]")
            except Exception as e:
                console.print(f"[yellow]无法计算参考声纹：{e}，已跳过。[/yellow]")

    seg_paths = step_preprocess(raw_files, encoder_obj=encoder_obj, ref_emb=ref_emb)
    if not seg_paths:
        console.print("[red]切片结果为空，请检查音频文件或切片参数。[/red]")
        return

    results = step_transcribe(seg_paths)
    step_finalize(results, seg_paths)

    if Confirm.ask("是否删除临时下载文件(temp_dir)以释放磁盘空间？", default=False):
        shutil.rmtree(CFG["temp_dir"], ignore_errors=True)
        console.print("[dim]临时文件已清除。[/dim]")


if __name__ == "__main__":
    main()
