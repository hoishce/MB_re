#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Encoder helper CLI

Runs in a separate process to safely load the project encoder on CPU
and compute embeddings / similarities without contaminating the caller
process's DLL search path.

Usage:
  # compute reference embedding
  python encoder_cli.py --model <path> --ref ref.wav

  # compute averaged reference embedding from a manifest
  python encoder_cli.py --model <path> --ref-manifest refs.jsonl

  # compute similarities from a JSONL manifest
  python encoder_cli.py --model <path> --ref ref.wav --manifest voiceprint_input.jsonl

  # legacy multi-target mode
  python encoder_cli.py --model <path> --ref ref.wav --targets a.wav b.wav c.wav

Outputs JSON to stdout. Success payloads always include one result per
input row in ``results``; legacy compatibility keeps a ``sims`` alias.
"""
from pathlib import Path
import sys
import os
import json
import argparse
import traceback
from contextlib import redirect_stdout

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def path_key(p):
    return os.path.normcase(Path(p).resolve().as_posix())


def read_jsonl(path: str | Path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_embedding(encoder, audio_path: Path):
    wav = encoder.preprocess_wav(str(audio_path))
    emb = encoder.embed_utterance(wav)
    return emb


def main():
    parser = argparse.ArgumentParser(description='Encoder helper CLI')
    parser.add_argument('--model', type=str, help='Encoder checkpoint file or directory')
    parser.add_argument('--ref', type=str, help='Reference WAV (16k)')
    parser.add_argument('--ref-manifest', type=str, help='JSONL manifest with reference paths to average')
    parser.add_argument('--manifest', type=str, help='JSONL manifest with segment_id and path')
    parser.add_argument('--targets', nargs='*', help='Target WAV files to compare (optional)')
    parser.add_argument('--threshold', type=float, default=0.62, help='Informational pass threshold for output status')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Inference device')
    args = parser.parse_args()
    if not args.ref and not args.ref_manifest:
        parser.error('either --ref or --ref-manifest is required')

    try:
        # Prefer repository root on sys.path
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        # Best-effort: adjust DLL ordering before importing torch/encoder
        try:
            import ensure_dlls
            ensure_dlls.ensure_dll_priority()
        except Exception:
            pass
        try:
            import check_cudnn_paths
        except Exception:
            pass

        # Now import encoder module and keep its normal prints off stdout so stdout stays valid JSON.
        from models.encoder import inference as encoder

        with redirect_stdout(sys.stderr):
            # Load model (CPU)
            model_path = args.model or str(repo_root / 'saved_models' / 'encoder' / 'saved_models' / 'pretrained.pt')
            if args.device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                device = 'cpu'
            elif args.device == 'cuda':
                device = 'cuda'
            else:
                try:
                    import torch

                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except Exception:
                    device = 'cpu'
            from pathlib import Path as _P
            mp = _P(model_path)
            if not encoder.is_loaded():
                try:
                    encoder.load_model(mp, device=device)
                except Exception:
                    # try without device kwarg
                    encoder.load_model(mp)

            # compute reference embedding
            ref_embs = []
            if args.ref_manifest:
                for row in read_jsonl(args.ref_manifest):
                    ref_path = Path(str(row.get('path') or '')).expanduser()
                    if not ref_path.exists():
                        continue
                    ref_embs.append(load_embedding(encoder, ref_path))
            else:
                ref_path = Path(args.ref)
                if not ref_path.exists():
                    raise FileNotFoundError(f"ref not found: {ref_path}")
                ref_embs.append(load_embedding(encoder, ref_path))

            if not ref_embs:
                raise RuntimeError("no valid reference embeddings")

            import numpy as _np
            ref_arrs = [_np.asarray(emb, dtype=float) for emb in ref_embs]
            ref_mean = _np.mean(_np.stack(ref_arrs, axis=0), axis=0)
            ref_list = [float(x) for x in ref_mean.tolist()]

            # If no targets, return embedding
            if not args.manifest and not args.targets:
                payload = {'status': 'ok', 'emb': ref_list}
            else:
                # Compute similarity for each target row and always emit one result per input
                out = []
                import numpy as _np
                r = _np.array(ref_list, dtype=float)
                rows = []
                if args.manifest:
                    rows = read_jsonl(args.manifest)
                else:
                    rows = [{"segment_id": Path(t).stem, "path": t} for t in (args.targets or [])]

                for row in rows:
                    segment_id = str(row.get("segment_id") or "").strip() or Path(str(row.get("path") or "")).stem
                    target_path = Path(str(row.get("path") or ""))
                    target = target_path.resolve().as_posix() if str(target_path) else ""
                    target_key = path_key(target_path) if str(target_path) else ""
                    try:
                        if not target_path.exists():
                            out.append({
                                "segment_id": segment_id,
                                "target": target,
                                "target_key": target_key,
                                "ok": False,
                                "sim": None,
                                "error": "not found",
                                "voiceprint_status": "voiceprint_reject",
                            })
                            continue
                        twav = encoder.preprocess_wav(str(target_path))
                        temb = encoder.embed_utterance(twav)
                        v = float(_np.dot(r, temb) / (_np.linalg.norm(r) * _np.linalg.norm(temb) + 1e-10))
                        out.append({
                            "segment_id": segment_id,
                            "target": target,
                            "target_key": target_key,
                            "ok": True,
                            "sim": v,
                            "error": None,
                            "voiceprint_status": "voiceprint_pass" if v >= float(args.threshold) else "voiceprint_reject",
                        })
                    except Exception as e:
                        out.append({
                            "segment_id": segment_id,
                            "target": target,
                            "target_key": target_key,
                            "ok": False,
                            "sim": None,
                            "error": f"{type(e).__name__}: {e}",
                            "voiceprint_status": "voiceprint_reject",
                        })
                payload = {"status": "ok", "results": out, "sims": out}

        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({'status':'error', 'error': str(e), 'trace': tb}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
