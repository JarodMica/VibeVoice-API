from pathlib import Path

import torch

import soundfile as sf
from vibevoice.infer_api import VibeVoiceInferencer

model = VibeVoiceInferencer(model_path="Jmica/Vibevoice7B")
audio, sr = model.generate_tts(
    script="hello there, what's for breakfast today?",
    voice_sample_path="demo/voices/en-Alice_woman.wav",
    num_speakers=1,
)

# soundfile expects float32/float64 so cast the model output before writing
waveform = audio.squeeze(0).to(torch.float32).cpu().numpy()
out_path = Path("hello_output.wav")
sf.write(out_path, waveform, sr)  # write mono wav
print(f"Saved audio to {out_path.resolve()}")
