"""Standalone voice trigger test: listen for 'capture image', beep on detection."""

import json
import math
import os
import struct
import subprocess
import tempfile
import time
import wave

import pyaudio
import vosk

RATE = 16000
CHANNELS = 1
CHUNK = 4000  # ~250ms at 16kHz
FORMAT = pyaudio.paInt16

TRIGGER_PHRASE = "capture image"
COOLDOWN_SECONDS = 3.0

ALSA_OUTPUT_DEVICE = os.environ.get("ALSA_DEVICE", "plughw:2,0")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-en-us-0.15")


def find_emeet_device_index(pa: pyaudio.PyAudio) -> int | None:
    """Scan PyAudio input devices for the EMEET mic. Returns device index or None."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info.get("name", "")
        if info.get("maxInputChannels", 0) > 0 and ("EMEET" in name or "M0" in name):
            print(f"  Found EMEET mic: index={i}, name={name!r}")
            return i
    return None


def generate_beep_wav(path: str, freq: int = 880, duration: float = 0.15):
    """Write a short sine-wave beep to a WAV file."""
    n_samples = int(RATE * duration)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        for i in range(n_samples):
            sample = int(24000 * math.sin(2 * math.pi * freq * i / RATE))
            wf.writeframes(struct.pack("<h", sample))


def play_beep():
    """Generate a beep WAV and play it through the speaker."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        generate_beep_wav(path)
        subprocess.run(
            ["aplay", "-D", ALSA_OUTPUT_DEVICE, path],
            capture_output=True,
            timeout=5,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"ERROR: Vosk model not found at {MODEL_PATH}")
        print("Download from: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        return

    vosk.SetLogLevel(-1)
    model = vosk.Model(MODEL_PATH)
    recognizer = vosk.KaldiRecognizer(model, RATE)

    pa = pyaudio.PyAudio()
    device_index = find_emeet_device_index(pa)

    stream_kwargs = {
        "format": FORMAT,
        "channels": CHANNELS,
        "rate": RATE,
        "input": True,
        "frames_per_buffer": CHUNK,
    }
    if device_index is not None:
        stream_kwargs["input_device_index"] = device_index
    else:
        print("  EMEET mic not found, using default input device")

    stream = pa.open(**stream_kwargs)
    print(f"\nListening for '{TRIGGER_PHRASE}'... (Ctrl+C to stop)\n")

    last_trigger = 0.0
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print(f"  [final] {text}")
                if TRIGGER_PHRASE in text and (time.time() - last_trigger) > COOLDOWN_SECONDS:
                    print(f"  >>> TRIGGERED on final: '{text}'")
                    play_beep()
                    last_trigger = time.time()
            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "")
                if text:
                    print(f"  [partial] {text}", end="\r")
                if TRIGGER_PHRASE in text and (time.time() - last_trigger) > COOLDOWN_SECONDS:
                    print(f"\n  >>> TRIGGERED on partial: '{text}'")
                    play_beep()
                    last_trigger = time.time()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
