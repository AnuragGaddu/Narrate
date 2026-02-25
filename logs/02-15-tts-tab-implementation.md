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

---

## 7. Auto-Speaking VLM Output on Tab 1

### 7.1 Problem

Tab 1 (Narration Camera) generated VLM descriptions and played TTS through
the **browser's `<audio>` element**. This meant audio came out of whatever
device the browser was using, not the Pi's USB speaker. For the product to
work as a standalone device (speaker and Pi operating independently of a
browser), all audio must be server-side.

### 7.2 Changes to `capture_route()`

Removed the TTS synthesis block from `POST /capture`. Previously it
synthesized a WAV, saved it to `static/audio/`, and returned an `audio_url`
for the browser to play. Now it only returns `{"text": "..."}` after VLM
inference. TTS is handled separately via `/tts_play`.

### 7.3 Changes to Tab 1 JavaScript

- Removed the `<audio>` element entirely — no browser-based audio playback.
- Created a shared `playViaSpeaker(text)` JS function that calls
  `POST /tts_play` (server-side Piper synthesis + `aplay` through USB speaker).
- **Capture button**: After receiving VLM text from `/capture`, displays the
  text immediately, then automatically calls `playViaSpeaker()` so the
  description is read aloud without any extra click.
- **Speak button**: Now calls `playViaSpeaker()` to replay the last narration
  through the USB speaker, instead of the old browser-audio `/speak` endpoint.
- Added a `<span id="speaker-status">` element showing "Speaking..." / "Done."
  / error messages.

### 7.4 Audio flow after changes

```
Capture click
  → POST /capture (VLM describes frame, returns text)
  → Text displayed in UI immediately
  → POST /tts_play (Piper synthesizes WAV → aplay → USB speaker)
  → Status shows "Speaking..." then "Done."

Speak click (replay)
  → POST /tts_play with last narration text
  → Same server-side playback path
```

All audio goes through `aplay -D plughw:2,0` on the Pi. The browser is only
a control panel — closing it mid-playback does not stop the sound.

---

## 8. Debugging — TTS 503 "Not Available" Error

### 8.1 Symptom

After deploying the Tab 1 auto-speak changes, the web UI showed a red
"TTS not available" error after every capture. The Flask terminal showed:

```
POST /tts_play HTTP/1.1" 503 -
POST /tts_play HTTP/1.1" 503 -
POST /tts_play HTTP/1.1" 503 -
```

The 503 comes from `tts_play_route()` when `tts.is_available()` returns False.

### 8.2 Hypotheses

- **A**: `PIPER_AVAILABLE` is False — `from piper import PiperVoice` failed
  at module load time.
- **B**: The `.onnx` voice model file path doesn't resolve correctly.
- **C**: `PiperVoice.load()` throws an exception caught by the bare `except`.
- **D**: The app is running outside the venv, so `piper` isn't importable.

### 8.3 Instrumentation

Added debug logs to `tts.py` at:

1. The `except ImportError` block (log the import error).
2. `_ensure_loaded()` when `PIPER_AVAILABLE` is False.
3. `_ensure_loaded()` at the onnx path existence check.
4. `_ensure_loaded()` when `PiperVoice.load()` throws.

### 8.4 Evidence from debug logs

```json
{"location":"tts.py:import","message":"piper import FAILED","data":{"error":"No module named 'piper'"},"hypothesisId":"A"}
{"location":"tts.py:_ensure_loaded","message":"PIPER_AVAILABLE is False","data":{},"hypothesisId":"A"}
```

**Hypothesis A confirmed.** Piper could not be imported because the user ran
`python app.py` without activating the virtual environment. The system Python
(`/usr/bin/python3.13`) does not have `piper-tts` installed — it is only in
`.venv`.

### 8.5 Verification

```bash
# System Python — no piper
$ /usr/bin/python -c "from piper import PiperVoice"
ModuleNotFoundError: No module named 'piper'

# Venv Python — piper available
$ source .venv/bin/activate && python -c "from piper import PiperVoice; print('OK')"
OK
```

Confirmed by checking the running process:

```
$ readlink -f /proc/24077/exe
/usr/bin/python3.13
```

The Flask process was using the system Python, not the venv.

### 8.6 Resolution

Always activate the venv before running the app:

```bash
source .venv/bin/activate && python app.py
```

---

## 9. Debugging — VLM Unavailable After Venv Activation

### 9.1 Symptom

After fixing the TTS issue by activating the venv, the VLM stopped working:

```
WARNING:vlm:hailo_platform.genai not installed – VLM unavailable
```

### 9.2 Root cause

The Hailo SDK (`hailo_platform`, `hailo_platform.genai`) is installed as a
**system package** (via apt/dpkg), not a pip package. The venv was created
with `include-system-site-packages = false` (the default), which blocked
access to system-level packages.

This created a conflict:

- **Without venv**: Hailo works, Piper doesn't (no `piper-tts`).
- **With venv**: Piper works, Hailo doesn't (no `hailo_platform`).

### 9.3 Fix

Changed `.venv/pyvenv.cfg`:

```
# Before
include-system-site-packages = false

# After
include-system-site-packages = true
```

This allows the venv to see both its own packages (piper-tts, flask, etc.)
and system packages (hailo_platform, hailo_platform.genai).

### 9.4 Verification

```bash
$ source .venv/bin/activate
$ python -c "from piper import PiperVoice; print('piper OK')"
piper OK
$ python -c "from hailo_platform.genai import VLM; print('hailo OK')"
hailo OK
```

Both libraries are now accessible in the same Python process.

### 9.5 Lesson

When working on a Raspberry Pi with hardware SDKs installed system-wide
(Hailo, rpicam, etc.), always create venvs with `--system-site-packages`:

```bash
python -m venv --system-site-packages .venv
```

Or fix an existing venv by editing `pyvenv.cfg` as shown above.

---

## 10. Updated Implementation Summary

### Files changed (cumulative for the day)

| File | Change |
|------|--------|
| `tts.py` | Fixed `synthesize` → `synthesize_wav` for Piper v1.4.x API; fixed wave open mode to `"wb"`. |
| `app.py` | Added tabbed UI (Narration Camera + Text to Speech); added `POST /tts_play` route; updated Tab 1 to auto-speak VLM output through USB speaker via `/tts_play`; removed browser `<audio>` playback from Tab 1. |
| `.venv/pyvenv.cfg` | Changed `include-system-site-packages` from `false` to `true` so both venv and system packages (Hailo SDK) are accessible. |

### System dependencies added

| Package | Purpose |
|---------|---------|
| `espeak-ng` | Grapheme-to-phoneme conversion (required by Piper) |
| `libespeak-ng-dev` | Development headers for espeakbridge compilation |
| `piper-tts==1.4.1` | Upgraded from 1.4.0; includes pre-built `espeakbridge.so` for aarch64 |

### Key takeaway: Running the app

The app **must** be started with the venv activated:

```bash
cd ~/Narrate
source .venv/bin/activate
python app.py
```

This ensures both `piper-tts` (venv package) and `hailo_platform` (system
package) are available to the same Python process.

---

## 11. Bug Fixes — Tab 1 Button State Management

### 11.1 Bug 1: Speak button stuck disabled after failed capture

**Symptom**: If a capture failed (network error or empty VLM response), the
Speak button stayed permanently disabled, even though the user had valid
narration from a previous successful capture stored in `window._lastNarration`.

**Root cause**: The Capture handler disabled Speak at the top
(`speakBtn.disabled = true`) but only re-enabled it inside the
`if (data.text)` success branch. The `else` (VLM error) and `catch`
(network error) branches never touched Speak, leaving it disabled.

### 11.2 Bug 2: No concurrency guard between Capture and Speak

**Symptom**: The Capture and Speak handlers each independently toggled
`captureBtn.disabled` and `speakBtn.disabled`. If one handler's
`playViaSpeaker()` finished and re-enabled buttons while the other was still
running, both buttons could become active during active TTS playback,
allowing concurrent `aplay` calls to the same speaker.

**Root cause**: Button state was managed with scattered `disabled = true/false`
assignments in each handler with no shared coordination.

### 11.3 Fix: Centralized `lockButtons()` / `unlockButtons()` with `_busy` guard

Replaced the individual `disabled` assignments with three shared helpers:

```javascript
let _busy = false;

function lockButtons() {
    _busy = true;
    captureBtn.disabled = true;
    speakBtn.disabled = true;
}

function unlockButtons() {
    _busy = false;
    captureBtn.disabled = false;
    speakBtn.disabled = !window._lastNarration;
}
```

Both handlers now follow the same pattern:

```javascript
if (_busy) return;
lockButtons();
// ... await async work ...
unlockButtons();
```

This fixes both bugs:

- **Bug 1**: `unlockButtons()` is called at the end of every code path
  (success, error, catch). It sets `speakBtn.disabled = !window._lastNarration`,
  so if a previous capture stored valid narration, Speak stays enabled even
  after a subsequent capture fails.
- **Bug 2**: The `_busy` flag prevents either handler from starting while the
  other is active, making concurrent `aplay` calls impossible.
