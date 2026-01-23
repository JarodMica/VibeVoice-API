"""
Object-oriented interface for loading VibeVoice models and running TTS inference.
"""
import numpy as np
import random
import torch

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union


from transformers import set_seed

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

SAMPLE_RATE = 24000

def set_seeds(seed=0):
    if seed==-1:
        seed_value = random.randint(0, 2**32 - 1)
    else:
        seed_value = seed
    set_seed(seed_value)

class VibeVoiceInferencer:
    """
    Convenience wrapper around the processor + model so callers can reuse loaded
    weights and just invoke `generate_tts` on the object.
    """

    def __init__(
        self,
        model_path: str = "Jmica/VibeVoice7B",
        *,
        device: str = "auto",
        ddpm_steps: int = 10,
    ) -> None:
        """
        Args:
            model_path: HF repo id or local checkpoint path.
            device: "auto" prefers CUDA -> MPS -> CPU. Explicit strings supported as well.
            ddpm_steps: Number of diffusion inference steps for subsequent generations.
        """

        self.device = _resolve_device(device)
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)
        self.model = _load_model(model_path, self.device)
        self.model.set_ddpm_inference_steps(num_steps=ddpm_steps)

    def generate_tts(
        self,
        *,
        script: str,
        voice_sample_path: Union[str, Path, Sequence[Union[str, Path]]],
        num_speakers: int = 1,
        cfg_scale: float = 1.3,
        disable_cloning: bool = False,
        seed=-1
    ) -> Tuple[torch.Tensor, int]:
        """
        Run inference and return a waveform tensor along with the sample rate.

        Args:
            script: Dialogue/podcast text. Plain lines are auto-tagged as "Speaker X:".
            voice_sample_path: Path (or list of paths) to .wav files for voice cloning.
            num_speakers: Number of speakers to cycle through when tagging the script.
            cfg_scale: Classifier-free guidance scale.
            disable_cloning: Skip supplying the voice sample (prefill disabled).
            seed: set seed (-1 is random)
        """
        set_seeds(seed)

        if num_speakers < 1:
            raise ValueError("num_speakers must be >= 1")

        formatted_script = _format_script(script, num_speakers)

        processor_kwargs = {
            "text": [formatted_script],
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        if not disable_cloning:
            voice_samples = _coerce_voice_samples(voice_sample_path)
            processor_kwargs["voice_samples"] = [voice_samples]

        inputs = self.processor(**processor_kwargs)
        inputs = _move_inputs_to_device(inputs, self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            is_prefill=not disable_cloning,
        )

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise RuntimeError("Model returned no audio. Try adjusting cfg_scale or your prompt.")

        return outputs.speech_outputs[0], SAMPLE_RATE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_voice_samples(
    voice_sample_path: Union[str, Path, Sequence[Union[str, Path]]]
) -> List[str]:
    if isinstance(voice_sample_path, (str, Path)):
        paths: Iterable[Union[str, Path]] = [voice_sample_path]
    else:
        paths = voice_sample_path
    resolved = [str(Path(p)) for p in paths]
    for path in resolved:
        if not Path(path).exists():
            raise FileNotFoundError(f"Voice sample not found: {path}")
    return resolved


def _format_script(raw_text: str, speaker_count: int) -> str:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    formatted: List[str] = []
    rotation = 0
    for line in lines:
        if line.lower().startswith("speaker ") and ":" in line:
            formatted.append(line)
        else:
            speaker_id = rotation % speaker_count
            formatted.append(f"Speaker {speaker_id}: {line}")
            rotation += 1
    return "\n".join(formatted)


def _resolve_device(requested: str) -> str:
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if req == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but not available on this machine.")
    return req


def _select_dtype_and_attn(device: str) -> Tuple[torch.dtype, str]:
    if device == "cuda":
        return torch.bfloat16, "flash_attention_2"
    if device == "mps":
        return torch.float32, "sdpa"
    return torch.float32, "sdpa"


def _load_model(
    model_path: str, device: str
) -> VibeVoiceForConditionalGenerationInference:
    torch_dtype, attn_impl = _select_dtype_and_attn(device)
    kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_impl,
    }
    if device == "cuda":
        kwargs["device_map"] = "cuda"
    elif device == "cpu":
        kwargs["device_map"] = "cpu"
    else:  # mps
        kwargs["device_map"] = None

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        **kwargs,
    )
    if device == "mps":
        model.to("mps")
    model.eval()
    return model


def _move_inputs_to_device(inputs, device: str):
    for key, value in inputs.items():
        if torch.is_tensor(value):
            target = device if device in ("cuda", "mps") else "cpu"
            inputs[key] = value.to(target)
    return inputs


__all__ = ["VibeVoiceInferencer", "SAMPLE_RATE"]

