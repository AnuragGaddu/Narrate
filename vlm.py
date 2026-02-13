"""Vision-Language Model module for image-to-text narration.

Supports Qwen2-VL on Hailo-10H (Hailo GenAI API).
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
            return "[Qwen2-VL on Hailo-10H: integration pending.]"
        # TODO: Run inference on Hailo NPU
        return ""


def get_vlm() -> VLMEngine:
    """Get VLM engine (Hailo Qwen2-VL)."""
    return HailoQwen2VL()


# Singleton
_vlm_engine: Optional[VLMEngine] = None


def get_vlm_engine() -> VLMEngine:
    global _vlm_engine
    if _vlm_engine is None:
        _vlm_engine = get_vlm()
    return _vlm_engine
