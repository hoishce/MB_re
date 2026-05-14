from pathlib import Path
import sys
# Ensure repository root is on sys.path and prioritize env DLLs early
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    import ensure_dlls
    ensure_dlls.ensure_dll_priority()
except Exception:
    # best-effort: continue if ensure_dlls is not available
    pass
def _require_compatible_numpy() -> None:
    """Fail fast when the environment has an incompatible NumPy release."""
    version = np.__version__.split("+", 1)[0]
    if version != "1.26.4":
        raise RuntimeError(
            f"Detected NumPy {np.__version__}, but this project expects NumPy 1.26.4. "
            "Please reinstall dependencies from requirements.txt and keep the pinned "
            "version (numpy==1.26.4)."
        )


import os
import torch
import numpy as np
from models.encoder import inference as encoder
from models.synthesizer.inference import Synthesizer
from models.vocoder.hifigan import inference as vocoder
from utils.argutils import print_args
import soundfile as sf
import torchaudio # Added for audio resampling
from pydub import AudioSegment
from typing import Optional
import argparse

_require_compatible_numpy()

# Configuration
# You need to download pre-trained models and place them in the specified paths.
# Refer to the MockingBird GitHub repository (https://github.com/babysor/MockingBird) for model download links and setup instructions.
def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


encoder_model_path = _env_path(
    "MOCKINGBIRD_ENCODER_CKPT",
    repo_root / "saved_models" / "encoder" / "saved_models" / "pretrained.pt",
)
# Synthesizer checkpoint dir (may contain mandarin.pt when download completes)
synthesizer_model_path = _env_path(
    "MOCKINGBIRD_SYNTHESIZER_CKPT",
    repo_root / "saved_models" / "synthesizer" / "saved_models" / "mandarin",
)
vocoder_model_path = _env_path(
    "MOCKINGBIRD_VOCODER_CKPT",
    repo_root / "saved_models" / "vocoder" / "saved_models" / "pretrained" / "pretrained.pt",
)

# Input audio for voice cloning (e.g., a short WAV file of the target voice)
# This file can now be in various formats (mp3, wav, flac, etc.).
clone_audio_path = _env_path("MOCKINGBIRD_REFERENCE_AUDIO", repo_root / "references" / "demo.wav")

# Text to be synthesized
text_to_synthesize = "你好"

def convert_audio_for_mockingbird(input_audio_path: Path, output_dir: Path = Path("temp_audio")) -> Path:
    """
    Converts an audio file to 16kHz, mono, WAV format suitable for Mockingbird.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_audio_path = output_dir / f"{input_audio_path.stem}_converted.wav"

    print(f"Converting audio: {input_audio_path} to {output_audio_path}")
    audio = AudioSegment.from_file(input_audio_path)
    audio = audio.set_frame_rate(encoder.sampling_rate) # Set to 16kHz
    audio = audio.set_channels(1) # Set to mono
    audio.export(output_audio_path, format="wav")
    print("Audio conversion complete.")
    return output_audio_path


def _resolve_checkpoint(path: Path) -> Optional[Path]:
    """
    If `path` is a file, return it. If it's a directory, search recursively for a .pt or .pth
    file and return the first reasonable candidate. Returns None if nothing found.
    """
    if path is None:
        return None
    if path.exists() and path.is_file():
        return path
    if path.exists() and path.is_dir():
        # Preference order: pretrained, encoder, synthesizer, g_ (hifigan naming), first found
        candidates = []
        for ext in ("*.pt", "*.pth"):
            candidates.extend(list(path.rglob(ext)))
        if not candidates:
            return None
        # prefer by name
        for name_key in ("pretrained", "encoder", "synthesizer", "g_", "vocoder"):
            for c in candidates:
                if name_key in c.name.lower():
                    return c
        return candidates[0]
    return None

def main():
    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU. Performance may be significantly slower.")
        # Optionally, you might want to exit or warn more strongly if GPU is expected.
        # return

    # Load models
    print("Loading models...")
    # Resolve checkpoints if directories were provided
    enc_ckpt = _resolve_checkpoint(encoder_model_path)
    syn_ckpt = _resolve_checkpoint(synthesizer_model_path)
    voc_ckpt = _resolve_checkpoint(vocoder_model_path)

    if enc_ckpt is None:
        print(f"Encoder checkpoint not found under {encoder_model_path}. Please point to a .pt/.pth file.")
        return
    if syn_ckpt is None:
        print(f"Synthesizer checkpoint not found under {synthesizer_model_path}. Please point to a .pt/.pth file.")
        return
    if voc_ckpt is None:
        print(f"Vocoder checkpoint not found under {vocoder_model_path}. Please point to a .pt/.pth file.")
        return

    print(f"Using encoder checkpoint: {enc_ckpt}")
    print(f"Using synthesizer checkpoint: {syn_ckpt}")
    print(f"Using vocoder checkpoint: {voc_ckpt}")

    # Ensure models are loaded to the correct device
    encoder.load_model(enc_ckpt, device=device)
    synthesizer = Synthesizer(syn_ckpt)
    # hifigan.load_model signature doesn't take a `device` kwarg.
    # Try to resolve a local hifigan config JSON inside the repo and pass it explicitly.
    repo_root = Path(__file__).resolve().parents[1]
    hifigan_config = repo_root / "models" / "vocoder" / "hifigan" / "config_16k_.json"
    if hifigan_config.exists():
        vocoder.load_model(voc_ckpt, config_fpath=hifigan_config)
    else:
        vocoder.load_model(voc_ckpt)
    print("Models loaded.")

    # Clear CUDA cache for better memory management on GPU
    if device == "cuda":
        torch.cuda.empty_cache()

    # 1. Prepare reference audio (convert if necessary and generate embedding)
    print(f"Preparing reference audio: {clone_audio_path}")
    converted_audio_path = convert_audio_for_mockingbird(clone_audio_path)

    original_wav, sampling_rate = sf.read(converted_audio_path)
    original_wav = torch.from_numpy(original_wav).float()

    # After pydub conversion, it should already be 16kHz, but we keep this check for robustness
    if sampling_rate != encoder.sampling_rate:
        print(f"Warning: Converted audio is {sampling_rate}Hz, resampling to {encoder.sampling_rate}Hz again.")
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=encoder.sampling_rate).to(device)
        original_wav = resampler(original_wav.to(device)).cpu().numpy()
    else:
        original_wav = original_wav.numpy()

    # Get the speaker embedding from the reference audio
    preprocessed_wav = encoder.preprocess_wav(original_wav)
    embed = encoder.embed_utterance(preprocessed_wav)
    print("Speaker embedding generated.")

    # 2. Synthesize spectrogram from text and embedding
    print(f"Synthesizing text: \"{text_to_synthesize}\"")
    # Synthesizer expects numpy embeddings
    specs = synthesizer.synthesize_spectrograms([text_to_synthesize], [embed.astype(np.float32)])
    spectrogram = specs[0]
    print("Spectrogram synthesized.")

    # 3. Vocode spectrogram to waveform
    print("Vocoding spectrogram...")
    # Vocode using the vocoder implementation (expects a numpy mel matrix)
    vocoder_result = vocoder.infer_waveform(spectrogram)
    if isinstance(vocoder_result, tuple) and len(vocoder_result) >= 2:
        generated_wav, output_sample_rate = vocoder_result
    else:
        generated_wav = vocoder_result
        output_sample_rate = getattr(synthesizer, "sample_rate", encoder.sampling_rate)
    print("Waveform generated.")

    # Save the output audio
    output_audio_path = "output_clone.wav"
    sf.write(output_audio_path, generated_wav.astype(np.float32), output_sample_rate)
    print(f"Cloned audio saved to {output_audio_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MockingBird demo with optional checkpoint paths")
    parser.add_argument("--enc", "-e", type=str, help="Encoder checkpoint file or directory")
    parser.add_argument("--syn", "-s", type=str, help="Synthesizer checkpoint file or directory")
    parser.add_argument("--voc", "-v", type=str, help="Vocoder checkpoint file or directory")
    parser.add_argument("--audio", "-a", type=str, help="Reference audio file to use")
    parser.add_argument("--text", "-t", type=str, help="Text to synthesize (overrides default)")
    parser.add_argument("--dry-run", action="store_true", help="Only show resolved checkpoint paths and exit")
    args = parser.parse_args()

    if args.enc:
        encoder_model_path = Path(args.enc)
    if args.syn:
        synthesizer_model_path = Path(args.syn)
    if args.voc:
        vocoder_model_path = Path(args.voc)
    if args.audio:
        clone_audio_path = Path(args.audio)
    if args.text:
        text_to_synthesize = args.text

    if args.dry_run:
        print("Resolved encoder checkpoint:", _resolve_checkpoint(encoder_model_path))
        print("Resolved synthesizer checkpoint:", _resolve_checkpoint(synthesizer_model_path))
        print("Resolved vocoder checkpoint:", _resolve_checkpoint(vocoder_model_path))
    else:
        main()
