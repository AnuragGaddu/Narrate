"""Piper TTS module for text-to-speech narration output."""

import io
import os
import tempfile
import wave

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    PiperVoice = None


# Default voice model path - user can override via env PIPER_MODEL
DEFAULT_MODEL = os.environ.get(
    "PIPER_MODEL",
    os.path.join(os.path.dirname(__file__), "voices", "en_US-lessac-medium")
)


class TTSEngine:
    """Text-to-speech engine using Piper."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or DEFAULT_MODEL
        self._voice = None
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """Load Piper voice model if not already loaded."""
        if not PIPER_AVAILABLE:
            return False
        if self._loaded and self._voice is not None:
            return True
        # Piper needs .onnx path; .onnx.json is alongside
        onnx_path = self.model_path
        if not onnx_path.endswith(".onnx"):
            onnx_path = f"{self.model_path}.onnx"
        if not os.path.exists(onnx_path):
            return False
        try:
            self._voice = PiperVoice.load(onnx_path)
            self._loaded = True
            return True
        except Exception:
            return False

    def synthesize_to_file(self, text: str, output_path: str) -> bool:
        """Synthesize text to a WAV file. Returns True on success."""
        if not text or not text.strip():
            return False
        if not self._ensure_loaded():
            return False
        try:
            with wave.open(output_path, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            return True
        except Exception:
            return False

    def synthesize_to_bytes(self, text: str) -> bytes | None:
        """Synthesize text to WAV bytes. Returns None on failure."""
        if not text or not text.strip():
            return None
        if not self._ensure_loaded():
            return None
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            buffer.seek(0)
            return buffer.read()
        except Exception:
            return None

    def is_available(self) -> bool:
        """Check if TTS is available and model is loaded."""
        return self._ensure_loaded()


# Singleton for app use
_tts_engine: TTSEngine | None = None


def get_tts() -> TTSEngine:
    """Get or create the TTS engine singleton."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine
