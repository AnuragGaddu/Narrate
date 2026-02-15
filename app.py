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

from flask import Flask, Response, jsonify, render_template_string, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# rpicam-vid subprocess for stream; reader thread updates _last_jpeg
_rpicam_process = None
_camera_ready = False
_last_jpeg = None
_last_jpeg_lock = threading.Lock()

# When set, video_feed serves this frame instead of live stream for 3 seconds
_frozen_jpeg = None
_frozen_until = 0.0
_freeze_lock = threading.Lock()

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
    """MJPEG stream endpoint. Serves frozen captured frame for 3s after capture, else live stream."""
    def gen():
        global _frozen_jpeg
        while _camera_ready:
            with _freeze_lock:
                now = time.time()
                if _frozen_jpeg is not None and now >= _frozen_until:
                    _frozen_jpeg = None  # expire freeze
                if _frozen_jpeg is not None:
                    frame = _frozen_jpeg
                else:
                    frame = None
            if frame is None:
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
    <title>Narrate</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { margin-bottom: 16px; }

        /* Tab bar */
        .tab-bar { display: flex; gap: 4px; margin-bottom: 20px; }
        .tab-btn {
            padding: 10px 24px; font-size: 15px; border: none; border-radius: 8px 8px 0 0;
            cursor: pointer; background: #16213e; color: #94a3b8; transition: background 0.2s;
        }
        .tab-btn.active { background: #0f0f1a; color: #eee; }
        .tab-panel { display: none; }
        .tab-panel.active { display: block; }

        /* Tab 1 - Narration Camera */
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
        .speaker-status { font-size: 14px; color: #94a3b8; margin-top: 8px; }
        .speaker-status.success { color: #4ade80; }
        .speaker-status.error { color: #f87171; }

        /* Tab 2 - Text to Speech */
        .tts-container { max-width: 640px; }
        .tts-container textarea {
            width: 100%; min-height: 160px; padding: 14px; font-size: 16px; font-family: inherit;
            border: 2px solid #2a2a4a; border-radius: 8px; background: #16213e; color: #eee;
            resize: vertical; outline: none; transition: border-color 0.2s;
        }
        .tts-container textarea:focus { border-color: #60a5fa; }
        .tts-controls { margin-top: 12px; display: flex; align-items: center; gap: 16px; }
        .tts-status { font-size: 14px; color: #94a3b8; }
        .tts-status.error { color: #f87171; }
        .tts-status.success { color: #4ade80; }
    </style>
</head>
<body>
    <h1>Narrate</h1>

    <div class="tab-bar">
        <button class="tab-btn active" data-tab="camera">Narration Camera</button>
        <button class="tab-btn" data-tab="tts">Text to Speech</button>
    </div>

    <!-- Tab 1: Narration Camera -->
    <div id="tab-camera" class="tab-panel active">
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
        <p id="speaker-status" class="speaker-status"></p>
    </div>

    <!-- Tab 2: Text to Speech -->
    <div id="tab-tts" class="tab-panel">
        <div class="tts-container">
            <textarea id="tts-input" placeholder="Type or paste text here..."></textarea>
            <div class="tts-controls">
                <button id="tts-speak" class="btn-speak">Speak</button>
                <span id="tts-status" class="tts-status"></span>
            </div>
        </div>
    </div>

    <script>
        /* ---- Tab switching ---- */
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
            });
        });

        /* ---- Tab 1: Narration Camera ---- */
        const captureBtn = document.getElementById('capture');
        const speakBtn = document.getElementById('speak');
        const narrationEl = document.getElementById('narration');
        const speakerStatus = document.getElementById('speaker-status');

        captureBtn.addEventListener('click', async () => {
            captureBtn.disabled = true;
            speakBtn.disabled = true;
            narrationEl.textContent = 'Processing...';
            narrationEl.className = 'loading';
            speakerStatus.textContent = '';
            try {
                const res = await fetch('/capture', { method: 'POST' });
                const data = await res.json();
                if (data.text) {
                    narrationEl.textContent = data.text;
                    narrationEl.className = '';
                    speakBtn.disabled = false;
                    window._lastNarration = data.text;
                    if (data.speaker === 'ok') {
                        speakerStatus.textContent = 'Played on speaker.';
                        speakerStatus.className = 'speaker-status success';
                    } else if (data.speaker && data.speaker !== 'not_attempted') {
                        speakerStatus.textContent = 'Speaker error: ' + data.speaker;
                        speakerStatus.className = 'speaker-status error';
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
            speakerStatus.textContent = 'Playing on speaker...';
            speakerStatus.className = 'speaker-status';
            try {
                const res = await fetch('/speak', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    speakerStatus.textContent = 'Played on speaker.';
                    speakerStatus.className = 'speaker-status success';
                } else {
                    speakerStatus.textContent = data.error || 'Playback failed.';
                    speakerStatus.className = 'speaker-status error';
                }
            } catch (e) {
                speakerStatus.textContent = 'Error: ' + e.message;
                speakerStatus.className = 'speaker-status error';
            }
            speakBtn.disabled = false;
        });

        /* ---- Tab 2: Text to Speech ---- */
        const ttsInput = document.getElementById('tts-input');
        const ttsSpeakBtn = document.getElementById('tts-speak');
        const ttsStatus = document.getElementById('tts-status');

        ttsSpeakBtn.addEventListener('click', async () => {
            const text = ttsInput.value.trim();
            if (!text) {
                ttsStatus.textContent = 'Please enter some text first.';
                ttsStatus.className = 'tts-status error';
                return;
            }
            ttsSpeakBtn.disabled = true;
            ttsStatus.textContent = 'Synthesizing and playing...';
            ttsStatus.className = 'tts-status';
            try {
                const res = await fetch('/tts_play', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    ttsStatus.textContent = 'Done.';
                    ttsStatus.className = 'tts-status success';
                } else {
                    ttsStatus.textContent = data.error || 'Playback failed.';
                    ttsStatus.className = 'tts-status error';
                }
            } catch (e) {
                ttsStatus.textContent = 'Error: ' + e.message;
                ttsStatus.className = 'tts-status error';
            }
            ttsSpeakBtn.disabled = false;
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
    """Capture frame, run VLM, optionally generate TTS. Freezes stream to this frame for 3s."""
    global _frozen_jpeg, _frozen_until
    frame = capture_frame()
    if frame is None:
        return jsonify({"error": "Camera not available"}), 500

    # Show captured frame on the stream area for 3 seconds
    try:
        from PIL import Image
        import io
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        with _freeze_lock:
            _frozen_jpeg = buf.getvalue()
            _frozen_until = time.time() + 3.0
    except Exception as e:
        logger.debug("Could not freeze frame for display: %s", e)

    try:
        vlm = get_vlm()
        text = vlm.describe_image(frame)
        if not text:
            text = "Could not describe image."

        speaker_status = "not_attempted"
        tts = get_tts()
        if tts.is_available():
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                if tts.synthesize_to_file(text, wav_path):
                    ok, err = _play_wav_on_speaker(wav_path)
                    speaker_status = "ok" if ok else err
                else:
                    speaker_status = "synthesis_failed"
            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        return jsonify({"text": text, "speaker": speaker_status})
    except Exception as e:
        logger.exception("Capture failed")
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak_route():
    """Synthesize TTS for given text and play through the Pi speaker."""
    data = request.get_json()
    text = (data or {}).get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tts = get_tts()
    if not tts.is_available():
        return jsonify({"error": "TTS not available"}), 503

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        if not tts.synthesize_to_file(text, wav_path):
            return jsonify({"error": "TTS synthesis failed"}), 500

        ok, err = _play_wav_on_speaker(wav_path)
        if not ok:
            return jsonify({"error": err}), 500
        return jsonify({"status": "ok"})
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ALSA device for the USB speaker (EMEET OfficeCore M0 Plus, card 2)
ALSA_DEVICE = os.environ.get("ALSA_DEVICE", "plughw:2,0")


def _play_wav_on_speaker(wav_path: str) -> tuple[bool, str]:
    """Play a WAV file through the Pi's USB speaker via aplay.

    Returns (success, error_message). On success error_message is empty.
    """
    try:
        result = subprocess.run(
            ["aplay", "-D", ALSA_DEVICE, wav_path],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:500]
            logger.error("aplay failed: %s", stderr)
            return False, f"Audio playback failed: {stderr}"
        return True, ""
    except subprocess.TimeoutExpired:
        logger.error("aplay timed out")
        return False, "Audio playback timed out"
    except Exception as e:
        logger.exception("aplay error")
        return False, str(e)


@app.route("/tts_play", methods=["POST"])
def tts_play_route():
    """Synthesize text and play it through the USB speaker."""
    data = request.get_json()
    text = (data or {}).get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tts = get_tts()
    if not tts.is_available():
        return jsonify({"error": "TTS not available"}), 503

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        if not tts.synthesize_to_file(text, wav_path):
            return jsonify({"error": "TTS synthesis failed"}), 500

        ok, err = _play_wav_on_speaker(wav_path)
        if not ok:
            return jsonify({"error": err}), 500
        return jsonify({"status": "ok"})
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def main():
    init_camera()
    if not _camera_ready:
        logger.warning("Camera not available - stream will be blank")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()