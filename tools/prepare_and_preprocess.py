import argparse
import re
from pathlib import Path
import soundfile as sf
import numpy as np
import importlib.util
from pathlib import Path as _Path

# load synthesizer audio module directly from file to avoid importing heavy submodules
_repo_root = _Path(__file__).resolve().parents[1]
_synth_audio_path = _repo_root / 'models' / 'synthesizer' / 'audio.py'
spec = importlib.util.spec_from_file_location('synth_audio', str(_synth_audio_path))
synth_audio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth_audio)

from models.synthesizer.hparams import hparams
from models.encoder import inference as encoder


def parse_transcript(transcript_path: Path):
    segments = []
    pattern = re.compile(r"\[(?P<s>[0-9]+\.?[0-9]*)s\s*->\s*(?P<e>[0-9]+\.?[0-9]*)s\]\s*(?P<t>.+)")
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                print("Warning: failed to parse transcript line:", line)
                continue
            s = float(m.group('s'))
            e = float(m.group('e'))
            t = m.group('t').strip()
            segments.append((s, e, t))
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True, help='Path to converted WAV (16kHz mono)')
    parser.add_argument('--transcript', required=True, help='Path to transcript file (timestamped lines)')
    parser.add_argument('--out-dataset', default='datasets/minidataset', help='Directory to write split wavs')
    parser.add_argument('--out-synth', default='saved_models/synth_minidataset', help='Synthesizer preprocess output dir')
    parser.add_argument('--encoder-ckpt', default=str(Path(__file__).resolve().parents[1] / 'saved_models' / 'encoder' / 'saved_models' / 'pretrained.pt'), help='Encoder checkpoint path for embeddings')
    args = parser.parse_args()

    audio_path = Path(args.audio)
    transcript_path = Path(args.transcript)
    out_dataset = Path(args.out_dataset)
    out_synth = Path(args.out_synth)
    encoder_ckpt = Path(args.encoder_ckpt)

    assert audio_path.exists(), f"Audio not found: {audio_path}"
    assert transcript_path.exists(), f"Transcript not found: {transcript_path}"

    segments = parse_transcript(transcript_path)
    if not segments:
        raise SystemExit('No segments parsed from transcript')

    # read audio
    audio, sr = sf.read(str(audio_path))
    if sr != hparams.sample_rate:
        print(f"Warning: audio sample rate {sr} != expected {hparams.sample_rate}. Resampling is recommended.")

    out_dataset.mkdir(parents=True, exist_ok=True)

    dict_info = {}
    for i, (s, e, text) in enumerate(segments, start=1):
        start_idx = int(round(s * sr))
        end_idx = int(round(e * sr))
        seg = audio[start_idx:end_idx]
        wav_name = f"utt_{i:03d}.wav"
        wav_path = out_dataset.joinpath(wav_name)
        sf.write(str(wav_path), seg, sr)
        dict_info[wav_name.rsplit('.', 1)[0]] = text
        print(f"Wrote {wav_path} -- {s}s to {e}s -> {text}")

    # Minimal synthesizer preprocessing (compute mel, save wav/mel/embed)
    out_synth.mkdir(parents=True, exist_ok=True)
    mels_dir = out_synth.joinpath('mels'); mels_dir.mkdir(exist_ok=True)
    audio_dir = out_synth.joinpath('audio'); audio_dir.mkdir(exist_ok=True)
    embeds_dir = out_synth.joinpath('embeds'); embeds_dir.mkdir(exist_ok=True)

    # Load encoder once
    try:
        encoder.load_model(encoder_ckpt, device='cpu')
    except TypeError:
        # older signature
        encoder.load_model(encoder_ckpt)

    metadata = []
    for wav_fpath in sorted(out_dataset.glob('*.wav')):
        base = wav_fpath.stem
        text = dict_info.get(base, '')

        # load wav (librosa inside synth_audio.load_wav)
        wav = synth_audio.load_wav(str(wav_fpath), hparams.sample_rate)
        # compute mel
        mel = synth_audio.melspectrogram(wav, hparams)
        mel_frames = mel.shape[1]

        sub_basename = f"{wav_fpath.name}_00"
        mel_name = f"mel-{sub_basename}.npy"
        wav_name_out = f"audio-{sub_basename}.npy"
        embed_name = f"embed-{sub_basename}.npy"

        np.save(mels_dir.joinpath(mel_name), mel.T, allow_pickle=False)
        np.save(audio_dir.joinpath(wav_name_out), wav, allow_pickle=False)

        # compute embed
        pre_wav = encoder.preprocess_wav(wav)
        embed = encoder.embed_utterance(pre_wav)
        np.save(embeds_dir.joinpath(embed_name), embed, allow_pickle=False)

        metadata.append((wav_name_out, mel_name, embed_name, len(wav), mel_frames, text))

    # write metadata file
    train_fpath = out_synth.joinpath('train.txt')
    with train_fpath.open('w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join(map(str, m)) + '\n')
    print('Wrote metadata to', train_fpath)

    print('Minimal preprocessing complete.')


if __name__ == '__main__':
    main()
