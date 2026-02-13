# Narrate (Narration Camera)

Project-level instructions for AI assistants working on this codebase.

## What this project does

**Narrate** is a Flask web app that turns a Raspberry Pi camera into an accessibility tool:

1. **Live stream** – MJPEG video feed from Pi Cam (rpicam-vid subprocess).
2. **Capture & describe** – User clicks "Capture"; the current frame is sent to a vision-language model (VLM) to produce a short, accessibility-oriented description.
3. **Speak** – Optional text-to-speech (Piper) for the description; audio is generated and played in the browser.

Target use: describing the scene in front of the camera for someone who cannot see it.

## Project layout

| File         | Role |
|--------------|------|
| `app.py`     | Flask app: camera init, capture loop, routes (`/`, `/video_feed`, `/capture`, `/speak`), inline HTML/JS/CSS. |
| `tts.py`     | Piper TTS: `TTSEngine` singleton, `get_tts()`. Voice path from `PIPER_MODEL` or default `voices/en_US-lessac-medium`. |
| `vlm.py`     | VLM: `VLMEngine` interface; `HailoQwen2VL` (Hailo-10H, placeholder); `BLIPCaptioner` (CPU fallback via Hugging Face). `get_vlm_engine()` returns the active engine. |
| `requirements.txt` | Flask, Pillow, piper-tts, numpy, opencv; optional: transformers, torch, accelerate for BLIP. System: rpicam-apps (rpicam-vid, rpicam-still). |

## Conventions and patterns

- **Python**: Standard library + type hints where used (`str | None`, `Image.Image | np.ndarray`). No strict style enforced; keep existing patterns in each file.
- **Camera**: Global `_camera`, `_frame_buffer`, `_last_jpeg`; background thread `_capture_loop()` encodes frames to JPEG (PIL or OpenCV). `capture_frame()` returns raw array for VLM.
- **Lazy init**: TTS and VLM are created on first use via `get_tts()` and `get_vlm()` in `app.py`, which delegate to `tts.get_tts` and `vlm.get_vlm_engine`.
- **Static audio**: TTS WAVs are written under `static/audio/` with UUID filenames; served at `/static/audio/<id>.wav`.

## Environment and runtime

- **Platform**: Intended for Raspberry Pi (Linux) with rpicam-apps (rpicam-vid, rpicam-still); camera init may fail if rpicam-vid is not in PATH.
- **Env**: `PIPER_MODEL` – path to Piper voice (directory or base path; code appends `.onnx` if needed).
- **Run**: `python app.py` → `init_camera()` (start rpicam-vid + MJPEG reader thread), then `app.run(host="0.0.0.0", port=5000)`.

## When editing

- **app.py**: Preserve threading (MJPEG reader loop, locks around `_last_jpeg`). Adding routes or changing HTML is fine; keep lazy TTS/VLM loading.
- **tts.py**: Piper expects `.onnx` (and `.onnx.json`) at `model_path`; `_ensure_loaded()` handles missing file by returning False.
- **vlm.py**: `describe_image()` accepts PIL `Image` or numpy array (RGB). BLIP uses `Salesforce/blip-image-captioning-base`. Hailo integration is stubbed; keep the same `VLMEngine` interface when implementing.

## Testing and dependencies

- Install: `pip install -r requirements.txt`.
- Pi: ensure rpicam-apps is installed (rpicam-vid, rpicam-still; same stack as rpicam-hello); optionally add Piper voice files under `voices/` or set `PIPER_MODEL`.
- Without GPU, BLIP runs on CPU (slower); Hailo path is not yet wired.

Use this file to stay consistent with project structure, conventions, and intent when suggesting or making changes.
