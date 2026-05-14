
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path so local imports (models, utils) resolve
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from models.encoder import inference as encoder
from models.synthesizer.inference import Synthesizer
from models.vocoder import inference as vocoder
from utils.argutils import print_args
from utils.model_utils import load_models
import soundfile as sf
import torchaudio # Added for audio resampling
from pydub import AudioSegment

# Configuration
# You need to download pre-trained models and place them in the specified paths.
# Refer to the MockingBird GitHub repository (https://github.com/babysor/MockingBird) for model download links and setup instructions.
encoder_model_path = Path("saved_models/default/encoder.pt")
synthesizer_model_path = Path("saved_models/default/synthesizer.pt")
vocoder_model_path = Path("saved_models/default/vocoder.pt")

# Input audio for voice cloning (e.g., a short WAV file of the target voice)
# This file can now be in various formats (mp3, wav, flac, etc.).
clone_audio_path = Path("demo_audio/reference.wav") # Replace with your reference audio file

# Text to be synthesized
text_to_synthesize = "你好，这是一个使用 Mockingbird 进行语音克隆的示例。"

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
    # Ensure models are loaded to the correct device
    encoder.load_model(encoder_model_path, device=device)
    synthesizer = Synthesizer(synthesizer_model_path).to(device)
    vocoder.load_model(vocoder_model_path, device=device)
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
    # Move embedding to device for synthesizer
    embed_on_device = torch.from_numpy(embed).to(device)
    specs = synthesizer.synthesize_spectrograms([text_to_synthesize], [embed_on_device])
    spectrogram = specs[0]
    print("Spectrogram synthesized.")

    # 3. Vocode spectrogram to waveform
    print("Vocoding spectrogram...")
    # Move spectrogram to device for vocoder
    spectrogram_on_device = torch.from_numpy(spectrogram).to(device)
    generated_wav = vocoder.infer_waveform(spectrogram_on_device, embed_on_device)
    print("Waveform generated.")

    # Save the output audio
    output_audio_path = "output_clone.wav"
    sf.write(output_audio_path, generated_wav.astype(np.float32), synthesizer.sample_rate)
    print(f"Cloned audio saved to {output_audio_path}")

if __name__ == "__main__":
    main()
