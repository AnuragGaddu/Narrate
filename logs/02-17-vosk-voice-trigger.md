# Voice Trigger Implementation (Vosk) — 2026-02-17

Full record of adding offline voice command support to the Narrate app, from
standalone prototype through integration into `app.py`. Saying "capture image"
now triggers the full capture-describe-speak pipeline hands-free.

---

## 1. Goal

Add a voice trigger so the user can say "capture image" to activate the
capture-VLM-TTS pipeline without touching the keyboard or browser. The
entire speech recognition pipeline must run locally on the Raspberry Pi
with no cloud dependency.

---

## 2. Technology Choice — Vosk

Vosk is an offline speech recognition toolkit based on Kaldi. Key properties:

- Runs entirely on-device (no network required).
- Lightweight English model (`vosk-model-small-en-us-0.15`, ~40 MB) suitable
  for keyword detection on a Raspberry Pi.
- Streams audio in real time via `KaldiRecognizer`.
- Returns JSON results with recognized text.

---

## 3. Dependencies

### 3.1 System packages

```bash
sudo apt-get install -y portaudio19-dev
```

`portaudio19-dev` provides the PortAudio library headers required to build
PyAudio's native extension.

### 3.2 Python packages

```bash
pip install vosk pyaudio
```

Installed versions:

| Package | Version |
|---------|---------|
| `vosk` | 0.3.45 |
| `pyaudio` | 0.2.14 |

### 3.3 Vosk model

```bash
cd models/
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip
```

This creates `models/vosk-model-small-en-us-0.15/` (~40 MB), which contains
the acoustic model, language model, and configuration files.

---

## 4. Hardware — Microphone Discovery

### 4.1 Listing ALSA capture devices

```
$ arecord -l

**** List of CAPTURE Hardware Devices ****
card 2: Plus [EMEET OfficeCore M0 Plus], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

The EMEET OfficeCore M0 Plus is a combined speaker + microphone USB device.
The microphone is on the same card as the speaker (card 2, device 0).

### 4.2 PyAudio device enumeration

When PyAudio initializes, it scans all ALSA devices. The EMEET mic was
found at PyAudio device index 1:

```
Found EMEET mic: index=1, name='EMEET OfficeCore M0 Plus: USB Audio (hw:2,0)'
```

The `_find_emeet_device_index()` function scans device names for "EMEET" or
"M0" with `maxInputChannels > 0` to auto-detect the correct index.

---

## 5. Standalone Test — `test_listener.py`

### 5.1 Purpose

Before integrating into `app.py`, a standalone test script was created to
verify that the microphone, Vosk model, and trigger phrase detection all
work correctly in isolation.

### 5.2 Implementation

```python
# Core loop structure
while True:
    data = stream.read(CHUNK, exception_on_overflow=False)

    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        if TRIGGER_PHRASE in text:
            play_beep()
    else:
        partial = json.loads(recognizer.PartialResult())
        text = partial.get("partial", "")
        if TRIGGER_PHRASE in text:
            play_beep()
```

Key design decisions:

- **Dual matching (final + partial)**: Vosk emits partial results as audio
  streams in and a final result when it detects a sentence boundary. Checking
  both reduces latency — the trigger fires on the partial result without
  waiting for a silence gap.
- **Substring matching**: `TRIGGER_PHRASE in text` rather than exact match.
  This handles natural variations like "please capture image" or "capture
  image now".
- **Cooldown timer**: 3-second cooldown prevents rapid re-fires from the
  same utterance appearing in both partial and final results.
- **Beep feedback**: A 880 Hz sine wave (0.15s duration) is generated
  programmatically via `math.sin` + `struct.pack` + `wave.open`, then played
  through the USB speaker via `aplay -D plughw:2,0`. No external sound files
  needed.

### 5.3 Audio configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 16000 Hz | Vosk models expect 16 kHz input |
| Channels | 1 (mono) | Speech recognition needs mono |
| Format | 16-bit signed LE | Standard PCM format for Vosk |
| Chunk size | 4000 frames | ~250 ms per read; balances latency vs. CPU |

### 5.4 Test results

```
Listening for 'capture image'... (Ctrl+C to stop)

  [partial] capture image
  >>> TRIGGERED on partial: 'capture image'
```

Mic streaming, speech recognition, trigger detection, and beep playback all
verified working.

---

## 6. Integration into `app.py`

### 6.1 Shared pipeline function

The capture route's inline logic was extracted into a standalone function
so both the Flask route and the voice listener can use the same code path:

```python
def _do_capture_and_narrate() -> tuple[str | None, str, str | None]:
    """Capture frame, describe via VLM, speak via TTS.
    Returns (description_text, speaker_status, error).
    """
```

The `/capture` route now calls this function, and the voice listener calls
it from its background thread.

### 6.2 Voice listener thread

```python
def _voice_listener_loop():
    """Background thread: listen for trigger phrase and run capture pipeline."""
```

Started as a daemon thread in `main()`:

```python
def _start_voice_listener():
    t = threading.Thread(target=_voice_listener_loop, daemon=True, name="vosk-listener")
    t.start()
```

The listener imports `pyaudio` and `vosk` inside the function body rather than
at module level. If either dependency is missing, the thread logs a warning
and exits — the rest of the app (Flask UI, camera, manual capture) continues
to work normally.

### 6.3 Concurrency guard — `_capture_lock`

A `threading.Lock` prevents concurrent pipeline executions:

```python
_capture_lock = threading.Lock()
```

The Flask `/capture` route acquires the lock with a 30-second timeout. The
voice listener uses non-blocking acquisition (`blocking=False`) and skips the
trigger if the lock is already held:

```python
if not _capture_lock.acquire(blocking=False):
    logger.info("Capture already in progress, skipping voice trigger")
    continue
```

This prevents two `aplay` processes from competing for the same speaker.

### 6.4 Echo suppression — `_voice_busy` Event

Without echo suppression, the TTS audio playing through the USB speaker
would be picked up by the adjacent USB microphone, and Vosk might
misinterpret the TTS output as a new "capture image" command.

```python
_voice_busy = threading.Event()
```

The listener sets the event before running the pipeline and clears it after:

```python
if triggered:
    _voice_busy.set()
    try:
        _play_beep()
        # ... run pipeline ...
    finally:
        _voice_busy.clear()
        recognizer = vosk.KaldiRecognizer(model, _VOSK_RATE)
        last_trigger = time.time()
```

While `_voice_busy` is set, the listener still reads audio chunks (to keep
the stream buffer from overflowing) but discards them:

```python
if _voice_busy.is_set():
    continue
```

After the pipeline completes, the `KaldiRecognizer` is recreated to flush
any buffered audio that accumulated during playback.

### 6.5 Trigger phrase and configuration

| Constant | Value | Purpose |
|----------|-------|---------|
| `_VOSK_TRIGGER` | `"capture image"` | Phrase that activates the pipeline |
| `_VOSK_COOLDOWN` | `3.0` seconds | Minimum time between triggers |
| `_VOSK_RATE` | `16000` Hz | Vosk model sample rate |
| `_VOSK_CHUNK` | `4000` frames | ~250 ms per audio read |
| `_VOSK_MODEL_PATH` | `models/vosk-model-small-en-us-0.15` | Relative to project dir |

---

## 7. Troubleshooting

### 7.1 Technology selection -- why Vosk

Several offline speech recognition options were evaluated:

| Library | Verdict | Reason |
|---------|---------|--------|
| **Vosk** | Chosen | Fully offline, no API keys, small model (~40 MB), good Python API, continuous recognition with keyword filtering |
| Picovoice Porcupine | Rejected | Requires API key for custom wake words; "capture image" is not a built-in keyword |
| PocketSphinx | Rejected | Less actively maintained, lower accuracy on modern hardware |
| openwakeword | Rejected | Requires custom model training for non-standard phrases |

Vosk was the only option that could detect an arbitrary phrase like "capture
image" out of the box, fully offline, with no account or API key.

### 7.2 ALSA/JACK warnings on PyAudio initialization

When the test script started, PyAudio dumped a wall of ALSA and JACK
warnings to stderr:

```
ALSA lib pcm_dsnoop.c:641:(snd_pcm_dsnoop_open) unable to open slave
ALSA lib pcm_dmix.c:1075:(snd_pcm_dmix_open) unable to open slave
ALSA lib pcm.c:2642:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.surround21
...
connect(2) call to /tmp/jack-1000/default/jack_0 failed (err=No such file or directory)
```

These are harmless -- PyAudio enumerates every possible ALSA and JACK
audio backend on startup, and most are not configured on a headless Pi.
The warnings go to stderr and initially obscured the script's actual stdout
output, making it look like something was broken.

### 7.3 Buffered stdout under `timeout`

When testing the script with a timeout wrapper:

```bash
timeout 10 python test_listener.py 2>&1
```

The script's own print statements ("Found EMEET mic", "Listening for...")
did not appear in the output. Only the ALSA warnings (stderr) were visible.

**Root cause:** Python buffers stdout when it detects the output is not a
terminal (which is the case when piped through `timeout`). The recognition
was actually working, but all output was stuck in the buffer.

**Fix:** Run with unbuffered output:

```bash
timeout 10 python -u test_listener.py 2>&1
```

The `-u` flag disables stdout buffering. After this change, the script's
messages appeared immediately alongside the ALSA noise, confirming the mic
was detected and Vosk was streaming audio.

### 7.4 Echo suppression design

**Problem:** The EMEET OfficeCore M0+ is a single USB device with both
speaker and microphone on the same card (card 2). When the TTS audio plays
through the speaker, the adjacent microphone picks it up. Vosk could then
misinterpret the TTS output as a new "capture image" command, causing an
infinite loop of captures.

**Failed approach considered:** Physically muting the mic during playback
via ALSA controls was considered but rejected -- it would require ALSA
mixer manipulation and might not take effect quickly enough.

**Solution implemented:** Software echo gate using `_voice_busy` Event:

1. Before the pipeline runs, set `_voice_busy`.
2. The listener loop continues reading audio chunks (to prevent buffer
   overflow) but discards them while the flag is set.
3. After the pipeline completes, clear `_voice_busy` and recreate the
   `KaldiRecognizer` to flush any audio that Vosk buffered during playback.
4. Reset `last_trigger` timestamp to enforce the cooldown period.

This approach is simple and reliable -- no ALSA mixer calls, no timing
dependencies, and the recognizer reset ensures no stale audio leaks through.

---

## 8. Files Changed

| File | Change |
|------|--------|
| `test_listener.py` | New file — standalone voice trigger test script (128 lines) |
| `app.py` | Added `_voice_listener_loop()`, `_find_emeet_device_index()`, `_generate_beep_wav()`, `_play_beep()`, `_start_voice_listener()`, `_do_capture_and_narrate()`. Added `_capture_lock` and `_voice_busy` threading primitives. Updated `main()` to start the listener. |
| `.gitignore` | (No change here — Vosk model is already under `models/` which was already ignored) |

### System dependencies added

| Package | Purpose |
|---------|---------|
| `portaudio19-dev` | System library for PyAudio native extension |
| `vosk` 0.3.45 | Offline speech recognition engine |
| `pyaudio` 0.2.14 | Python bindings for PortAudio (mic streaming) |

### Model downloaded

| File | Size | Source |
|------|------|--------|
| `models/vosk-model-small-en-us-0.15/` | ~40 MB | alphacephei.com/vosk/models |

---

## 9. End-to-End Test

```
$ source .venv/bin/activate
$ python app.py

INFO:__main__:Camera ready
INFO:__main__:Found EMEET mic: index=1, name='EMEET OfficeCore M0 Plus: USB Audio (hw:2,0)'
INFO:__main__:Voice listener active — say 'capture image' to trigger capture
 * Running on http://0.0.0.0:5000

(user says "capture image")

INFO:__main__:Voice trigger (partial): 'capture image'
INFO:__main__:Voice-triggered narration: The image shows a desk with a ... (speaker: ok)
```

The voice trigger fires, a beep confirms detection, the camera captures a
frame, the VLM describes it, and the description is spoken through the USB
speaker — all without touching the browser.
