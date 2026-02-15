"""Vision-Language Model module for image-to-text narration.

Supports Qwen2-VL on Hailo-10H (Hailo GenAI API).
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Narration prompt for accessibility
NARRATION_PROMPT = "Describe this image in one or two sentences for someone who cannot see it."

# Default HEF path relative to this file's directory
_DEFAULT_HEF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "Qwen2-VL-2B-Instruct.hef",
)


class VLMEngine:
    """Abstract VLM interface."""

    def describe_image(self, image: Image.Image | np.ndarray) -> str:
        """Return a text description of the image."""
        raise NotImplementedError

    def release(self) -> None:
        """Release any held resources."""
        pass


class HailoQwen2VL(VLMEngine):
    """Qwen2-VL on Hailo-10H via GenAI API.

    Requires the Hailo GenAI inference SDK (hailo_platform.genai)
    and models/Qwen2-VL-2B-Instruct.hef.
    """

    def __init__(self, hef_path: str | None = None):
        self.hef_path = hef_path or _DEFAULT_HEF
        self._vdevice = None
        self._vlm = None
        self._frame_shape = None   # (H, W, C) required by the model
        self._frame_dtype = None   # numpy dtype required by the model
        self._available = False

        if not os.path.exists(self.hef_path):
            logger.warning("HEF file not found at %s – VLM unavailable", self.hef_path)
            return

        try:
            from hailo_platform import VDevice
            from hailo_platform.genai import VLM

            logger.info("Initializing Hailo VDevice...")
            self._vdevice = VDevice()

            logger.info("Loading VLM from %s (this may take a moment)...", self.hef_path)
            self._vlm = VLM(self._vdevice, self.hef_path, optimize_memory_on_device=True)

            # Query the model's expected input frame format
            self._frame_shape = self._vlm.input_frame_shape()   # (H, W, C)
            self._frame_dtype = self._vlm.input_frame_format_type()

            self._available = True
            logger.info(
                "Hailo VLM ready – frame shape: %s, dtype: %s",
                self._frame_shape,
                self._frame_dtype,
            )
        except ImportError:
            logger.warning("hailo_platform.genai not installed – VLM unavailable")
        except Exception as exc:
            logger.error("Failed to initialize Hailo VLM: %s", exc, exc_info=True)
            # Clean up partial init
            self._cleanup()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def describe_image(self, image: Image.Image | np.ndarray) -> str:
        if not self._available or self._vlm is None:
            return "[Qwen2-VL on Hailo-10H: not available.]"

        try:
            # Convert PIL Image to numpy RGB array if needed
            if isinstance(image, Image.Image):
                image = np.array(image.convert("RGB"))

            # Resize / cast to what the model expects
            target_h, target_w, target_c = self._frame_shape
            if image.shape[0] != target_h or image.shape[1] != target_w:
                image = cv2.resize(
                    image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            if target_c == 1 and image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = np.expand_dims(image, axis=2)
            image = image.astype(self._frame_dtype)

            # Build structured prompt
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": NARRATION_PROMPT},
                    ],
                }
            ]

            # Run inference (non-streaming)
            result = self._vlm.generate_all(
                prompt=prompt,
                frames=[image],
                max_generated_tokens=150,
            )

            # Clear context so the next call starts fresh
            self._vlm.clear_context()

            return result.strip() if result else ""
        except Exception as exc:
            logger.error("VLM inference failed: %s", exc, exc_info=True)
            return "[VLM inference error.]"

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release Hailo VLM and VDevice resources."""
        self._cleanup()

    def _cleanup(self) -> None:
        if self._vlm is not None:
            try:
                self._vlm.release()
            except Exception:
                pass
            self._vlm = None
        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception:
                pass
            self._vdevice = None
        self._available = False


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
