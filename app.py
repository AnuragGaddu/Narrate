"""Narration Camera - Flask app with live Pi Cam stream, capture, and TTS."""

import atexit
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from flask import Flask, Response, jsonify, render_template_string, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# rpicam-vid subprocess for stream; reader thread updates _last_jpeg
_rpicam_process = None
_camera_ready = False
_last_jpeg = None
_last_jpeg_lock = threading.Lock()

# JPEG markers for MJPEG frame splitting
_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"

_DEBUG_LOG = "/home/anurag-gaddu/Narrate/.cursor/debug.log"

def _dbg(msg, data=None, hid=None):
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG), exist_ok=True)
        with open(_DEBUG_LOG, "a") as f:
            f.write(json.dumps({"message": msg, "data": data or {}, "hypothesisId": hid, "timestamp": int(time.time() * 1000), "location": "app.py:capture_frame"}) + "\n")
    except Exception:
        pass


def _terminate_rpicam():
    """Terminate rpicam-vid subprocess on exit."""
    global _rpicam_process
    if _rpicam_process is not None and _rpicam_process.poll() is None:
        _rpicam_process.terminate()
        try:
            _rpicam_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _rpicam_process.kill()
        _rpicam_process = None


def init_camera():
    """Start rpicam-vid subprocess and MJPEG reader thread."""
    global _rpicam_process, _camera_ready
    if shutil.which("rpicam-vid") is None:
        logger.warning("rpicam-vid not found - camera unavailable")
        _camera_ready = False
        return
    try:
        _rpicam_process = subprocess.Popen(
            [
                "rpicam-vid",
                "-t", "0",
                "-n",
                "--codec", "mjpeg",
                "-o", "-",
                "--width", "640",
                "--height", "480",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        atexit.register(_terminate_rpicam)
        t = threading.Thread(target=_mjpeg_reader_loop, daemon=True)
        t.start()
        # Wait briefly for first frame so _camera_ready is set by reader
        for _ in range(50):
            time.sleep(0.1)
            with _last_jpeg_lock:
                if _last_jpeg is not None:
                    break
        _camera_ready = True
        logger.info("Camera initialized (rpicam-vid)")
    except Exception as e:
        logger.error("Camera init failed: %s", e)
        _terminate_rpicam()
        _camera_ready = False


def _mjpeg_reader_loop():
    """Background thread: read MJPEG from rpicam-vid stdout, split frames, store latest JPEG."""
    global _last_jpeg, _camera_ready, _rpicam_process
    buf = b""
    stream = _rpicam_process.stdout if _rpicam_process else None
    if stream is None:
        return
    try:
        while _rpicam_process is not None and _rpicam_process.poll() is None:
            chunk = stream.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(_JPEG_SOI)
                end = buf.find(_JPEG_EOI, start) if start != -1 else -1
                if start != -1 and end != -1:
                    end += 2
                    jpeg = buf[start:end]
                    buf = buf[end:]
                    with _last_jpeg_lock:
                        _last_jpeg = jpeg
                else:
                    if len(buf) > 2 * 1024 * 1024:
                        buf = buf[-1024:]
                    break
    except Exception as e:
        logger.debug("MJPEG reader error: %s", e)
    finally:
        _camera_ready = False


def video_feed():
    """MJPEG stream endpoint."""
    def gen():
        while _camera_ready:
            with _last_jpeg_lock:
                frame = _last_jpeg
            if frame:
                yield (b"--FRAME\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.05)
    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=FRAME",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


def capture_frame():
    """Capture a single frame for VLM. Prefers latest frame from stream (avoids Pi camera exclusive-access conflict)."""
    # #region agent log
    _dbg("capture_frame entered", {"rpicam_vid_running": _rpicam_process is not None and _rpicam_process.poll() is None}, "A")
    # #endregion
    from PIL import Image
    import numpy as np
    import io

    # Prefer stream frame: rpicam-vid holds camera, so rpicam-still cannot run simultaneously
    with _last_jpeg_lock:
        jpeg = _last_jpeg
    if jpeg is not None:
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            arr = np.array(img)
            # #region agent log
            _dbg("capture_frame success from stream", {"shape": list(arr.shape)}, "A")
            # #endregion
            return arr
        except Exception as e:
            # #region agent log
            _dbg("stream frame decode failed", {"error": str(e)}, "A")
            # #endregion
            logger.warning("Stream frame decode failed: %s", e)

    # Fallback: rpicam-still (only works when rpicam-vid is not running)
    if shutil.which("rpicam-still") is None:
        # #region agent log
        _dbg("rpicam-still not found", {}, "E")
        # #endregion
        logger.error("rpicam-still not found")
        return None
    fd = None
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        fd = None
        result = subprocess.run(
            [
                "rpicam-still",
                "-n",
                "-o", path,
                "--width", "640",
                "--height", "480",
            ],
            capture_output=True,
            timeout=10,
        )
        # #region agent log
        stderr_s = result.stderr.decode(errors="replace")[:500] if result.stderr else ""
        _dbg("rpicam-still completed", {"returncode": result.returncode, "stderr": stderr_s}, "A")
        # #endregion
        if result.returncode != 0:
            # #region agent log
            _dbg("rpicam-still failed returncode", {"returncode": result.returncode, "stderr": stderr_s}, "H1")
            # #endregion
            logger.error("rpicam-still failed: %s", result.stderr.decode(errors="replace"))
            return None
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        # #region agent log
        _dbg("capture_frame success from rpicam-still", {"shape": list(arr.shape)}, "A")
        # #endregion
        return arr
    except subprocess.TimeoutExpired:
        # #region agent log
        _dbg("rpicam-still timed out", {}, "B")
        # #endregion
        logger.error("rpicam-still timed out")
        return None
    except Exception as e:
        # #region agent log
        _dbg("capture_frame exception", {"error": str(e)}, "H4")
        # #endregion
        logger.error("Capture failed: %s", e)
        return None
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


# Lazy init TTS and VLM
_tts = None
_vlm = None


def get_tts():
    global _tts
    if _tts is None:
        from tts import get_tts as _get_tts
        _tts = _get_tts()
    return _tts


def get_vlm():
    global _vlm
    if _vlm is None:
        from vlm import get_vlm_engine
        _vlm = get_vlm_engine()
    return _vlm


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Narration Camera</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { margin-bottom: 16px; }
        .stream-container { background: #0f0f1a; border-radius: 8px; overflow: hidden; margin-bottom: 16px; max-width: 640px; }
        .stream-container img { width: 100%; display: block; }
        .controls { margin-bottom: 16px; display: flex; gap: 12px; flex-wrap: wrap; }
        button { padding: 12px 24px; font-size: 16px; border: none; border-radius: 8px; cursor: pointer; }
        .btn-capture { background: #4ade80; color: #1a1a2e; }
        .btn-speak { background: #60a5fa; color: #1a1a2e; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .narration-box { background: #16213e; padding: 16px; border-radius: 8px; max-width: 640px; min-height: 80px; }
        .narration-box p { margin: 0; line-height: 1.5; }
        .loading { color: #94a3b8; }
        .error { color: #f87171; }
        audio { margin-top: 12px; width: 100%; max-width: 400px; }
    </style>
</head>
<body>
    <h1>Narration Camera</h1>
    <div class="stream-container">
        <img src="/video_feed" alt="Live camera stream" width="640" height="480">
    </div>
    <div class="controls">
        <button id="capture" class="btn-capture">Capture</button>
        <button id="speak" class="btn-speak" disabled>Speak</button>
    </div>
    <div class="narration-box">
        <p id="narration">Press Capture to describe what the camera sees.</p>
    </div>
    <audio id="audio" controls style="display:none"></audio>

    <script>
        const captureBtn = document.getElementById('capture');
        const speakBtn = document.getElementById('speak');
        const narrationEl = document.getElementById('narration');
        const audioEl = document.getElementById('audio');

        captureBtn.addEventListener('click', async () => {
            captureBtn.disabled = true;
            narrationEl.textContent = 'Processing...';
            narrationEl.className = 'loading';
            try {
                const res = await fetch('/capture', { method: 'POST' });
                const data = await res.json();
                if (data.text) {
                    narrationEl.textContent = data.text;
                    narrationEl.className = '';
                    speakBtn.disabled = false;
                    window._lastNarration = data.text;
                    if (data.audio_url) {
                        audioEl.src = data.audio_url;
                        audioEl.style.display = 'block';
                        audioEl.play();
                    }
                } else {
                    narrationEl.textContent = data.error || 'Failed to get description';
                    narrationEl.className = 'error';
                }
            } catch (e) {
                narrationEl.textContent = 'Error: ' + e.message;
                narrationEl.className = 'error';
            }
            captureBtn.disabled = false;
        });

        speakBtn.addEventListener('click', async () => {
            const text = window._lastNarration;
            if (!text) return;
            speakBtn.disabled = true;
            try {
                const res = await fetch('/speak', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                if (data.audio_url) {
                    audioEl.src = data.audio_url;
                    audioEl.style.display = 'block';
                    audioEl.play();
                }
            } catch (e) {
                console.error(e);
            }
            speakBtn.disabled = false;
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed_route():
    return video_feed()


@app.route("/capture", methods=["POST"])
def capture_route():
    """Capture frame, run VLM, optionally generate TTS."""
    frame = capture_frame()
    if frame is None:
        return jsonify({"error": "Camera not available"}), 500

    try:
        vlm = get_vlm()
        text = vlm.describe_image(frame)
        if not text:
            text = "Could not describe image."

        audio_url = None
        tts = get_tts()
        if tts.is_available():
            audio_id = str(uuid.uuid4())
            audio_dir = os.path.join(app.root_path, "static", "audio")
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f"{audio_id}.wav")
            if tts.synthesize_to_file(text, audio_path):
                audio_url = f"/static/audio/{audio_id}.wav"

        return jsonify({"text": text, "audio_url": audio_url})
    except Exception as e:
        logger.exception("Capture failed")
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak_route():
    """Generate TTS for given text."""
    data = request.get_json()
    text = (data or {}).get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tts = get_tts()
    if not tts.is_available():
        return jsonify({"error": "TTS not available"}), 503

    audio_id = str(uuid.uuid4())
    audio_dir = os.path.join(app.root_path, "static", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{audio_id}.wav")
    if tts.synthesize_to_file(text, audio_path):
        return jsonify({"audio_url": f"/static/audio/{audio_id}.wav"})
    return jsonify({"error": "TTS synthesis failed"}), 500


def main():
    init_camera()
    if not _camera_ready:
        logger.warning("Camera not available - stream will be blank")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
