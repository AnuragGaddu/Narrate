# TTS Tab Implementation Log — 2026-02-15

Full record of adding the "Text to Speech" tab to the Narrate app, including
hardware verification, dependency debugging, Piper API troubleshooting, and the
rationale for not using the Hailo AI HAT+ for TTS.

---

## 1. Goal

Add a second tab to the Flask GUI (`app.py`) with:

- A text box where the user types or pastes text.
- A "Speak" button that synthesizes speech locally with Piper TTS and plays it
  through the USB speaker (EMEET OfficeCore M0 Plus) connected to the
  Raspberry Pi.

---

## 2. Hardware — Verifying the USB Speaker

### 2.1 Listing ALSA playback devices

```
$ aplay -l

**** List of PLAYBACK Hardware Devices ****
card 0: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0
card 1: vc4hdmi1 [vc4-hdmi-1], device 0: MAI PCM i2s-hifi-0
card 2: Plus [EMEET OfficeCore M0 Plus], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

The EMEET is **card 2, device 0** → ALSA device name is `plughw:2,0`.

### 2.2 Confirming kernel recognition

```
$ cat /proc/asound/cards

 0 [vc4hdmi0       ]: vc4-hdmi - vc4-hdmi-0
 1 [vc4hdmi1       ]: vc4-hdmi - vc4-hdmi-1
 2 [Plus           ]: USB-Audio - EMEET OfficeCore M0 Plus
                      EMEET EMEET OfficeCore M0 Plus at usb-xhci-hcd.1-1, full speed
```

The speaker is connected at USB bus `xhci-hcd.1-1` in full-speed mode.

### 2.3 Testing audio output with `speaker-test`

```
$ speaker-test -D plughw:2,0 -c 1 -t sine -f 440 -l 1 -p 1

speaker-test 1.2.14
Playback device is plughw:2,0
Stream parameters are 48000Hz, S16_LE, 1 channels
Sine wave rate is 440.0000Hz
```

A 440 Hz tone played successfully through the EMEET speaker, confirming:

- The USB audio path works end-to-end.
- `plughw:2,0` is the correct ALSA device string.
- The speaker supports 48000 Hz, Signed 16-bit Little Endian, mono.

---

## 3. Piper TTS — Setup and Documentation

### 3.1 What is Piper?

Piper is a fast, local neural text-to-speech engine maintained by the
Open Home Foundation (formerly Rhasspy). Key facts:

- Runs ONNX neural network models on CPU via ONNX Runtime.
- Needs **espeak-ng** for grapheme-to-phoneme conversion.
- Voice models are `.onnx` files with a companion `.onnx.json` config.
- Produces 16-bit PCM WAV audio at the sample rate specified by the model
  (typically 22050 Hz for medium-quality voices).

### 3.2 Checking the installed version

```
$ pip show piper-tts

Name: piper-tts
Version: 1.4.0
Location: .venv/lib/python3.13/site-packages
Requires: onnxruntime
```

The library imported successfully:

```python
from piper import PiperVoice  # OK
```

### 3.3 Downloading a voice model

No voice model existed on disk. Downloaded the `en_US-lessac-medium` voice
(~61 MB ONNX + 4.8 KB JSON config) from HuggingFace:

```
$ mkdir -p voices
$ wget -O voices/en_US-lessac-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
$ wget -O voices/en_US-lessac-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Verified:

```
$ ls -lh voices/
total 61M
-rw-rw-r-- 1 anurag-gaddu anurag-gaddu  61M Feb 15 15:32 en_US-lessac-medium.onnx
-rw-rw-r-- 1 anurag-gaddu anurag-gaddu 4.8K Feb 15 15:32 en_US-lessac-medium.onnx.json
```

### 3.4 Confirming model loads

```python
from tts import get_tts
t = get_tts()
print('available:', t.is_available())  # True
```

The model loaded successfully via ONNX Runtime.

---

## 4. Debugging — TTS Synthesis Failures

### 4.1 First failure: `synthesize_to_file` returns False

After downloading the voice model, synthesis returned `False` with a 0-byte
WAV file. The `tts.py` code swallows exceptions in a bare `except`, so the
real error was hidden.

### 4.2 Root cause 1: Piper v1.4.x API change

The existing `tts.py` called:

```python
self._voice.synthesize(text, wav_file)
```

Inspecting the Piper v1.4.0 API revealed the signatures had changed:

```python
# Old API (pre-1.4):
#   synthesize(text, wav_file)  — wrote directly to wav_file

# New API (v1.4.x):
synthesize(text, syn_config=None, include_alignments=False)
    → Iterable[AudioChunk]   # returns audio chunks, does NOT accept wav_file

synthesize_wav(text, wav_file, syn_config=None, set_wav_format=True, ...)
    → Optional[list[PhonemeAlignment]]   # writes to wav_file
```

The old code was passing `wav_file` as the `syn_config` parameter, which
was silently ignored, producing no audio. The fix was to call
`synthesize_wav(text, wav_file)` instead.

### 4.3 Root cause 2: Missing `espeakbridge` native extension

Even after fixing the API call, synthesis raised:

```
ImportError: cannot import name 'espeakbridge' from 'piper'
```

The traceback:

```
piper/voice.py:285, in synthesize
    sentence_phonemes = self.phonemize(text)
piper/voice.py:233, in phonemize
    _ESPEAK_PHONEMIZER = EspeakPhonemizer(self.espeak_data_dir)
piper/phonemize_espeak.py:17, in __init__
    from . import espeakbridge
ImportError: cannot import name 'espeakbridge' from 'piper'
```

`espeakbridge` is a compiled C extension that wraps `libespeak-ng`. Two things
were missing:

1. **System library**: `espeak-ng` and `libespeak-ng-dev` were not installed.
2. **Pre-built wheel**: `piper-tts==1.4.0` did not include a pre-built
   `espeakbridge.so` for aarch64 — only the `.pyi` stub was present.

### 4.4 Fix: Install espeak-ng and upgrade piper-tts

```bash
# Install system dependencies
sudo apt-get install -y espeak-ng libespeak-ng-dev

# Upgrade to piper-tts 1.4.1 which ships the pre-built espeakbridge
# for manylinux_2_17_aarch64
pip install --force-reinstall --no-cache-dir piper-tts
```

After this, `piper-tts==1.4.1` was installed with the native
`espeakbridge.so` bundled in the wheel.

### 4.5 Fix: Update `tts.py` API calls

Changed both `synthesize_to_file` and `synthesize_to_bytes` in `tts.py`:

```python
# Before (broken with Piper >= 1.4):
self._voice.synthesize(text, wav_file)

# After (correct for Piper 1.4.x):
self._voice.synthesize_wav(text, wav_file)
```

Also changed `wave.open(output_path, "w")` → `wave.open(output_path, "wb")`
to match the binary mode expected by `synthesize_wav`.

### 4.6 Successful synthesis

```python
from piper import PiperVoice
import wave, os

v = PiperVoice.load('voices/en_US-lessac-medium.onnx')
with wave.open('/tmp/test_tts.wav', 'wb') as wf:
    v.synthesize_wav('Hello, this is a test of the text to speech system.', wf)

print('file size:', os.path.getsize('/tmp/test_tts.wav'))
# file size: 119852
```

### 4.7 Successful playback through USB speaker

```
$ aplay -D plughw:2,0 /tmp/test_tts.wav

Playing WAVE '/tmp/test_tts.wav' : Signed 16 bit Little Endian, Rate 22050 Hz, Mono
```

Audio came out of the EMEET OfficeCore M0 Plus successfully.

---

## 5. Why We Cannot Use the Hailo AI HAT+ for Piper TTS

### 5.1 What the Hailo AI HAT+ is

The Hailo AI HAT+ contains a Hailo-8L or Hailo-10H Neural Processing Unit
(NPU). It is designed to accelerate neural network inference — particularly
convolutional neural networks (CNNs) and transformer-based vision models. The
Narrate project already uses it for the Qwen2-VL vision-language model via
the Hailo GenAI API.

### 5.2 Why it cannot run Piper TTS

1. **Architecture mismatch**: Piper's neural network is a VITS-based
   (Variational Inference with adversarial learning for end-to-end
   Text-to-Speech) model. It contains operations like 1D convolutions,
   transposed convolutions, and flow-based decoder layers that are not
   in the set of operators supported by the Hailo compiler (DFC).

2. **No HEF compilation path**: To run a model on the Hailo NPU, it must
   be compiled into a HEF (Hailo Executable Format) file using the Hailo
   Dataflow Compiler. Hailo has not published a HEF for any TTS model,
   and the VITS architecture used by Piper would require significant
   operator support that does not exist in the current DFC.

3. **espeak-ng phonemizer dependency**: Piper's pipeline is not just the
   neural network — it first converts text to phonemes using espeak-ng
   (a CPU-based C library), then feeds phoneme IDs to the neural network.
   Even if the neural net could run on Hailo, the phonemization step must
   run on CPU.

4. **ONNX Runtime is fast enough**: On a Raspberry Pi 5, Piper synthesizes
   speech in well under real-time on CPU via ONNX Runtime. A short sentence
   synthesizes in ~2-3 seconds. There is no practical need for NPU
   acceleration.

### 5.3 Summary

The Hailo AI HAT+ accelerates **vision inference** (the Qwen2-VL model in
`vlm.py`). Piper TTS runs **locally on the CPU** via ONNX Runtime, which is
the correct and intended deployment for this type of model on Raspberry Pi.
Both run locally on the device with no cloud dependency.

---

## 6. Implementation Summary

### Files changed

| File      | Change |
|-----------|--------|
| `tts.py`  | Fixed `synthesize` → `synthesize_wav` for Piper v1.4.x API; fixed wave open mode to `"wb"`. |
| `app.py`  | Added tabbed UI (Narration Camera + Text to Speech); added `POST /tts_play` route that synthesizes and plays via `aplay -D plughw:2,0`. |

### System dependencies added

| Package             | Purpose |
|---------------------|---------|
| `espeak-ng`         | Grapheme-to-phoneme conversion (required by Piper) |
| `libespeak-ng-dev`  | Development headers for espeakbridge compilation |
| `piper-tts==1.4.1`  | Upgraded from 1.4.0; includes pre-built `espeakbridge.so` for aarch64 |

### Voice model downloaded

| File | Size | Source |
|------|------|--------|
| `voices/en_US-lessac-medium.onnx` | 61 MB | huggingface.co/rhasspy/piper-voices |
| `voices/en_US-lessac-medium.onnx.json` | 4.8 KB | huggingface.co/rhasspy/piper-voices |

### New endpoint

`POST /tts_play` — accepts `{"text": "..."}`, synthesizes WAV with Piper,
plays through USB speaker via `aplay -D plughw:2,0`, returns
`{"status": "ok"}`. ALSA device is configurable via `ALSA_DEVICE` env var.

### End-to-end test

```
$ curl -s -X POST http://127.0.0.1:5000/tts_play \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, this is a test of the text to speech tab."}'

{"status":"ok"}
```

Audio played through the EMEET OfficeCore M0 Plus USB speaker successfully.
