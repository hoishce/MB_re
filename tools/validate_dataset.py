import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    from models.synthesizer.hparams import hparams
except Exception:
    hparams = None


def validate_preprocessed(path: Path):
    train_f = path.joinpath('train.txt')
    if not train_f.exists():
        print('train.txt not found in', path)
        return 2

    lines = [l.strip() for l in train_f.read_text(encoding='utf-8').splitlines() if l.strip()]
    audio_dir = path.joinpath('audio')
    mels_dir = path.joinpath('mels')
    embeds_dir = path.joinpath('embeds')

    missing = False
    for i, l in enumerate(lines, start=1):
        parts = l.split('|')
        if len(parts) < 6:
            print(f'Bad metadata line {i}: {l}')
            missing = True
            continue
        audio_name, mel_name, embed_name = parts[0].strip(), parts[1].strip(), parts[2].strip()
        a = audio_dir.joinpath(audio_name)
        m = mels_dir.joinpath(mel_name)
        e = embeds_dir.joinpath(embed_name)
        if not a.exists():
            print('Missing audio:', a)
            missing = True
        if not m.exists():
            print('Missing mel:', m)
            missing = True
        if not e.exists():
            print('Missing embed:', e)
            missing = True
        # quick shape checks
        try:
            emb = np.load(e)
            if hparams is not None and emb.ndim == 1 and emb.shape[0] != hparams.speaker_embedding_size:
                print(f'Warning: embed {embed_name} size {emb.shape} != hparams.speaker_embedding_size {hparams.speaker_embedding_size}')
        except Exception as ex:
            print('Cannot load embed', e, ex)

    total = len(lines)
    print(f'Metadata entries: {total}.')
    if missing:
        print('Validation failed: missing files or bad metadata.')
        return 1
    print('Preprocessed dataset looks consistent.')
    return 0


def validate_raw_dataset(path: Path):
    wavs = sorted(path.glob('*.wav'))
    if not wavs:
        print('No .wav files found in', path)
        return 2
    bad = False
    for w in wavs:
        try:
            data, sr = sf.read(str(w))
            if hparams is not None and sr != hparams.sample_rate:
                print(f'Warning: {w.name} sample rate {sr} != expected {hparams.sample_rate}')
            chans = 1 if data.ndim == 1 else data.shape[1]
            dur = len(data) / float(sr)
            if dur < 0.4:
                print(f'Warning: {w.name} is very short ({dur:.2f}s)')
        except Exception as ex:
            print('Error reading', w, ex)
            bad = True
    if bad:
        return 1
    print('Raw wav dataset looks OK (sample rate warnings may be present).')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to preprocessed synth folder or raw wav folder')
    args = parser.parse_args()
    p = Path(args.path)
    if not p.exists():
        print('Path not found:', p)
        sys.exit(2)

    # determine type
    if p.joinpath('train.txt').exists():
        code = validate_preprocessed(p)
        sys.exit(code)
    else:
        code = validate_raw_dataset(p)
        sys.exit(code)


if __name__ == '__main__':
    main()
