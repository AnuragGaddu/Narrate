"""Vision-Language Model module for image-to-text narration.

Supports:
- Qwen2-VL on Hailo-10H (when Hailo GenAI API is available)
- CPU fallback: BLIP image captioning via Hugging Face
"""

import os
from typing import Optional

import numpy as np
from PIL import Image

# Narration prompt for accessibility
NARRATION_PROMPT = "Describe this image in one or two sentences for someone who cannot see it."


class VLMEngine:
    """Abstract VLM interface."""

    def describe_image(self, image: Image.Image | np.ndarray) -> str:
        """Return a text description of the image."""
        raise NotImplementedError


class HailoQwen2VL(VLMEngine):
    """Qwen2-VL on Hailo-10H via GenAI API.

    Requires Hailo GenAI inference SDK and Qwen2-VL-2B-Instruct.hef.
    Placeholder: integrate with your Hailo GenAI Python API when available.
    """

    def __init__(self, hef_path: str | None = None):
        self.hef_path = hef_path or os.path.join(
            os.path.expanduser("~"), "Downloads", "Qwen2-VL-2B-Instruct.hef"
        )
        self._model = None
        self._available = False
        # TODO: Load HEF and initialize Hailo GenAI inference when API is available
        if os.path.exists(self.hef_path):
            try:
                # Placeholder: Hailo GenAI Python API integration
                # from hailo_genai import ... 
                pass
            except ImportError:
                pass

    def describe_image(self, image: Image.Image | np.ndarray) -> str:
        if not self._available:
            return "[Qwen2-VL on Hailo-10H: integration pending. Using fallback.]"
        # TODO: Run inference on Hailo NPU
        return ""


class BLIPCaptioner(VLMEngine):
    """BLIP image captioning on CPU (Hugging Face). Lightweight fallback."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._available = False
        self._load()

    def _load(self):
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            model_id = "Salesforce/blip-image-captioning-base"
            self._processor = BlipProcessor.from_pretrained(model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(model_id)
            self._available = True
        except Exception:
            self._available = False

    def describe_image(self, image: Image.Image | np.ndarray) -> str:
        if not self._available:
            return "Unable to describe image: BLIP model not loaded."
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            out = self._model.generate(**inputs, max_length=50)
            return self._processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            return f"Error describing image: {e}"


def get_vlm(use_hailo: bool = False) -> VLMEngine:
    """Get VLM engine. Prefer Hailo if requested and available."""
    if use_hailo:
        hailo = HailoQwen2VL()
        if hailo._available:
            return hailo
    return BLIPCaptioner()


# Singleton
_vlm_engine: Optional[VLMEngine] = None


def get_vlm_engine() -> VLMEngine:
    global _vlm_engine
    if _vlm_engine is None:
        _vlm_engine = get_vlm(use_hailo=True)
    return _vlm_engine
