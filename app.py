"""Narration Camera - Flask dev portal with live Pi Cam stream, VLM, and TTS.

Voice trigger: say "capture image" to capture, describe, and narrate.
Dev portal shows real-time pipeline status via Server-Sent Events.
"""

import atexit
import json
import logging
import math
import multiprocessing
import os
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import ProcessPoolExecutor

from flask import Flask, Response, jsonify, render_template_string, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# SSE (Server-Sent Events) infrastructure
# ---------------------------------------------------------------------------

_sse_clients: list[queue.Queue] = []
_sse_clients_lock = threading.Lock()


def broadcast_event(event_type: str, data=None):
    """Push an SSE event to all connected browser clients."""
    payload = json.dumps({"type": event_type, "data": data or {}})
    dead: list[queue.Queue] = []
    with _sse_clients_lock:
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


class _SSELogHandler(logging.Handler):
    """Forward log records to SSE clients for UI display."""

    def emit(self, record):
        try:
            msg = self.format(record)
            broadcast_event("log", {"level": record.levelname.lower(), "message": msg})
        except Exception:
            pass


_sse_handler = _SSELogHandler()
_sse_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_sse_handler)

# ---------------------------------------------------------------------------
# Camera streaming (rpicam-vid MJPEG)
# ---------------------------------------------------------------------------

_rpicam_process = None
_camera_ready = False
_last_jpeg = None
_last_jpeg_lock = threading.Lock()

_frozen_jpeg = None
_frozen_until = 0.0
_freeze_lock = threading.Lock()

_capture_lock = threading.Lock()

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"

_DEBUG_LOG = "/home/anurag-gaddu/Narrate/.cursor/debug.log"


def _dbg(msg, data=None, hid=None):
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG), exist_ok=True)
        with open(_DEBUG_LOG, "a") as f:
            f.write(json.dumps({
                "message": msg, "data": data or {}, "hypothesisId": hid,
                "timestamp": int(time.time() * 1000), "location": "app.py",
            }) + "\n")
    except Exception:
        pass


def _terminate_rpicam():
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
                "rpicam-vid", "-t", "0", "-n",
                "--codec", "mjpeg", "-o", "-",
                "--width", "640", "--height", "480",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        atexit.register(_terminate_rpicam)
        t = threading.Thread(target=_mjpeg_reader_loop, daemon=True)
        t.start()
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
    """Background thread: read MJPEG from rpicam-vid stdout, split frames."""
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
    """MJPEG stream generator. Serves frozen frame for 3s after capture, else live."""
    def gen():
        global _frozen_jpeg
        while _camera_ready:
            with _freeze_lock:
                now = time.time()
                if _frozen_jpeg is not None and now >= _frozen_until:
                    _frozen_jpeg = None
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
    """Capture a single frame for VLM. Prefers stream frame, falls back to rpicam-still."""
    _dbg("capture_frame entered", {
        "rpicam_vid_running": _rpicam_process is not None and _rpicam_process.poll() is None,
    }, "A")
    from PIL import Image
    import numpy as np
    import io

    with _last_jpeg_lock:
        jpeg = _last_jpeg
    if jpeg is not None:
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            arr = np.array(img)
            _dbg("capture_frame success from stream", {"shape": list(arr.shape)}, "A")
            return arr, jpeg
        except Exception as e:
            _dbg("stream frame decode failed", {"error": str(e)}, "A")
            logger.warning("Stream frame decode failed: %s", e)

    if shutil.which("rpicam-still") is None:
        _dbg("rpicam-still not found", {}, "E")
        logger.error("rpicam-still not found")
        return None
    fd = None
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        fd = None
        result = subprocess.run(
            ["rpicam-still", "-n", "-o", path, "--width", "640", "--height", "480"],
            capture_output=True,
            timeout=10,
        )
        stderr_s = result.stderr.decode(errors="replace")[:500] if result.stderr else ""
        _dbg("rpicam-still completed", {"returncode": result.returncode, "stderr": stderr_s}, "A")
        if result.returncode != 0:
            _dbg("rpicam-still failed", {"returncode": result.returncode, "stderr": stderr_s}, "H1")
            logger.error("rpicam-still failed: %s", result.stderr.decode(errors="replace"))
            return None
        with open(path, "rb") as f:
            jpeg_fallback = f.read()
        img = Image.open(io.BytesIO(jpeg_fallback)).convert("RGB")
        arr = np.array(img)
        _dbg("capture_frame success from rpicam-still", {"shape": list(arr.shape)}, "A")
        return arr, jpeg_fallback
    except subprocess.TimeoutExpired:
        _dbg("rpicam-still timed out", {}, "B")
        logger.error("rpicam-still timed out")
        return None
    except Exception as e:
        _dbg("capture_frame exception", {"error": str(e)}, "H4")
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


# ---------------------------------------------------------------------------
# Lazy-init TTS and VLM singletons
# ---------------------------------------------------------------------------

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


_vlm_mp_ctx = multiprocessing.get_context("spawn")
_vlm_executor = ProcessPoolExecutor(max_workers=1, mp_context=_vlm_mp_ctx)


def _vlm_infer(frame_array):
    """Run VLM inference in a worker process (avoids GIL blocking main process)."""
    from vlm import get_vlm_engine
    vlm = get_vlm_engine()
    return vlm.describe_image(frame_array)


# ---------------------------------------------------------------------------
# Audio playback with stop support
# ---------------------------------------------------------------------------

ALSA_DEVICE = os.environ.get("ALSA_DEVICE", "plughw:2,0")

_aplay_process = None
_aplay_lock = threading.Lock()


def _play_wav_on_speaker(wav_path: str) -> tuple[bool, str]:
    """Play a WAV file via aplay. Tracks process so /stop_tts can kill it.

    Returns (success, error_message).
    """
    global _aplay_process
    try:
        with _aplay_lock:
            _aplay_process = subprocess.Popen(
                ["aplay", "-D", ALSA_DEVICE, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        proc = _aplay_process
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            with _aplay_lock:
                _aplay_process = None
            logger.error("aplay timed out")
            return False, "Audio playback timed out"

        with _aplay_lock:
            _aplay_process = None
        # -15 = SIGTERM from stop button — treat as intentional, not error
        if proc.returncode != 0 and proc.returncode != -15:
            stderr = proc.stderr.read().decode(errors="replace")[:500]
            logger.error("aplay failed: %s", stderr)
            return False, f"Audio playback failed: {stderr}"
        return True, ""
    except Exception as e:
        with _aplay_lock:
            _aplay_process = None
        logger.exception("aplay error")
        return False, str(e)


# ---------------------------------------------------------------------------
# Captured image + last narration storage
# ---------------------------------------------------------------------------

_captured_jpeg = None
_captured_jpeg_lock = threading.Lock()

_last_narration = None
_last_narration_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pipeline: capture -> VLM -> TTS (runs in background thread, broadcasts SSE)
# ---------------------------------------------------------------------------

_voice_busy = threading.Event()


def _run_pipeline(source: str = "manual"):
    """Full capture-VLM-TTS pipeline. Broadcasts SSE events at each stage."""
    global _frozen_jpeg, _frozen_until, _captured_jpeg, _last_narration

    _voice_busy.set()

    if not _capture_lock.acquire(blocking=False):
        broadcast_event("error", {"message": "Another capture is in progress"})
        logger.info("Pipeline skipped — already in progress (source: %s)", source)
        _voice_busy.clear()
        broadcast_event("trigger", {"active": False})
        broadcast_event("status", {"phase": "idle"})
        return

    try:
        # -- Stage 1: Capture --
        broadcast_event("status", {"phase": "capturing"})
        logger.info("Pipeline started (source: %s)", source)

        result = capture_frame()
        if result is None:
            broadcast_event("error", {"message": "Camera not available"})
            broadcast_event("status", {"phase": "idle"})
            return
        frame, jpeg_bytes = result

        # Store captured JPEG for the UI (using original JPEG, no re-encode needed)
        try:
            with _captured_jpeg_lock:
                _captured_jpeg = jpeg_bytes
            with _freeze_lock:
                _frozen_jpeg = jpeg_bytes
                _frozen_until = time.time() + 3.0
            broadcast_event("captured_image", {
                "url": "/captured_image?t=" + str(int(time.time() * 1000)),
            })
        except Exception as e:
            logger.debug("Could not freeze/store frame: %s", e)

        # -- Stage 2: VLM inference (runs in separate process to avoid GIL block) --
        broadcast_event("status", {"phase": "processing_vlm"})
        try:
            future = _vlm_executor.submit(_vlm_infer, frame)
            text = future.result(timeout=60)
            if not text:
                text = "Could not describe image."
        except Exception as e:
            broadcast_event("error", {"message": f"VLM failed: {e}"})
            broadcast_event("status", {"phase": "idle"})
            logger.exception("VLM inference failed")
            return

        broadcast_event("vlm_text", {"text": text})
        with _last_narration_lock:
            _last_narration = text

        # -- Stage 3: TTS synthesis + playback --
        broadcast_event("status", {"phase": "speaking"})
        try:
            tts = get_tts()
            if tts.is_available():
                fd, wav_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    if tts.synthesize_to_file(text, wav_path):
                        ok, err = _play_wav_on_speaker(wav_path)
                        if not ok:
                            broadcast_event("error", {"message": f"Speaker: {err}"})
                    else:
                        broadcast_event("error", {"message": "TTS synthesis failed"})
                finally:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
            else:
                broadcast_event("error", {"message": "TTS engine not available"})
        except Exception as e:
            broadcast_event("error", {"message": f"TTS error: {e}"})
            logger.exception("TTS failed")

        broadcast_event("status", {"phase": "idle"})
        logger.info("Pipeline complete (source: %s): %s", source, text[:80])
    finally:
        _capture_lock.release()
        _voice_busy.clear()
        broadcast_event("trigger", {"active": False})


# ---------------------------------------------------------------------------
# HTML / CSS / JS  —  Dev Portal (4-column layout)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Narrate — Dev Portal</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f0f1a;color:#e2e8f0;min-height:100vh}

/* Header */
header{background:#1a1a2e;padding:12px 24px;border-bottom:1px solid #2a2a4a;display:flex;align-items:center;gap:16px}
header h1{font-size:20px;font-weight:600}
.pipeline-badge{font-size:12px;padding:4px 14px;border-radius:12px;background:#16213e;color:#94a3b8;font-weight:500;transition:all .3s}
.pipeline-badge.active{background:#1e3a5f;color:#60a5fa}
.sse-dot{width:8px;height:8px;border-radius:50%;background:#dc2626;display:inline-block;margin-left:auto;flex-shrink:0}
.sse-dot.ok{background:#22c55e}

/* 4-column grid */
.portal{display:grid;grid-template-columns:180px 1fr 1fr 160px;gap:16px;padding:16px;height:calc(100vh - 52px);min-height:500px}

.col{display:flex;flex-direction:column;gap:12px;min-width:0}
.col h2{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:#64748b;padding-bottom:8px;border-bottom:1px solid #2a2a4a}

.card{background:#1a1a2e;border-radius:8px;border:1px solid #2a2a4a;overflow:hidden;display:flex;flex-direction:column}
.card-hdr{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:#64748b;padding:8px 12px;border-bottom:1px solid #2a2a4a;background:#16213e}
.card-body{padding:12px;flex:1}

/* Col 1 — Trigger */
.light-wrap{display:flex;flex-direction:column;align-items:center;gap:14px;padding:24px 12px}
.light{width:48px;height:48px;border-radius:50%;background:#dc2626;box-shadow:0 0 16px #dc262660;transition:all .3s}
.light.on{background:#22c55e;box-shadow:0 0 24px #22c55e60}
.light-label{font-size:13px;color:#94a3b8;text-align:center}
.btn-trig{width:100%;padding:10px;font-size:14px;font-weight:600;border:none;border-radius:6px;cursor:pointer;background:#4ade80;color:#0f0f1a;transition:opacity .2s}
.btn-trig:hover{opacity:.85}
.btn-trig:disabled{opacity:.4;cursor:not-allowed}

/* Col 2 — Video */
.vid-card img,.cap-card img{width:100%;display:block;background:#000}
.cap-empty{display:flex;align-items:center;justify-content:center;min-height:100px;color:#475569;font-size:13px}

/* Col 3 — Text */
.vlm-body{padding:14px;line-height:1.65;font-size:14px;min-height:80px;white-space:pre-wrap}
.vlm-body.ph{color:#475569}
.log-wrap{min-height:0;display:flex;flex-direction:column}
.log-scroll{overflow-y:auto;padding:6px 12px;font-family:'JetBrains Mono','Fira Code',monospace;font-size:11px;line-height:1.7;max-height:50vh}
.le{padding:1px 0}
.le.info{color:#94a3b8}
.le.warning{color:#fbbf24}
.le.error{color:#f87171}
.le.debug{color:#64748b}

/* Col 4 — TTS */
.tts-btns{display:flex;flex-direction:column;gap:8px}
.btn-r,.btn-s{width:100%;padding:10px;font-size:14px;font-weight:600;border:none;border-radius:6px;cursor:pointer;transition:opacity .2s}
.btn-r{background:#60a5fa;color:#0f0f1a}
.btn-s{background:#ef4444;color:#fff}
.btn-r:hover,.btn-s:hover{opacity:.85}
.btn-r:disabled,.btn-s:disabled{opacity:.4;cursor:not-allowed}
.tts-msg{font-size:12px;color:#94a3b8;text-align:center;margin-top:4px}

@media(max-width:900px){.portal{grid-template-columns:1fr;height:auto}}
</style>
</head>
<body>

<header>
  <h1>Narrate</h1>
  <span class="pipeline-badge" id="badge">Idle</span>
  <span class="sse-dot" id="sse" title="SSE disconnected"></span>
</header>

<div class="portal">

  <!-- Col 1: Trigger -->
  <div class="col">
    <h2>Trigger</h2>
    <div class="card">
      <div class="light-wrap">
        <div class="light" id="light"></div>
        <p class="light-label" id="light-label">Listening</p>
      </div>
    </div>
    <button class="btn-trig" id="trigBtn">Trigger</button>
  </div>

  <!-- Col 2: Video / Log -->
  <div class="col">
    <h2>Video</h2>
    <div class="card vid-card">
      <div class="card-hdr">Live Stream</div>
      <img src="/video_feed" alt="Live camera stream" id="vidStream">
    </div>
    <div class="card log-wrap">
      <div class="card-hdr">Log</div>
      <div class="log-scroll" id="log"></div>
    </div>
  </div>

  <!-- Col 3: Image / Caption -->
  <div class="col">
    <h2>Image / Caption</h2>
    <div class="card cap-card" id="capCard">
      <div class="card-hdr">Captured Image</div>
      <div class="cap-empty" id="capEmpty">No capture yet</div>
      <img id="capImg" src="" alt="" style="display:none">
    </div>
    <div class="card">
      <div class="card-hdr">VLM Output</div>
      <div class="vlm-body ph" id="vlm">Waiting for capture...</div>
    </div>
  </div>

  <!-- Col 4: TTS -->
  <div class="col">
    <h2>TTS</h2>
    <div class="tts-btns">
      <button class="btn-r" id="replayBtn" disabled>Replay</button>
      <button class="btn-s" id="stopBtn">Stop</button>
    </div>
    <p class="tts-msg" id="ttsMsg"></p>
  </div>

</div>

<script>
(function(){
  const $=id=>document.getElementById(id);
  const badge=$('badge'), sseDot=$('sse'),
        light=$('light'), lightLabel=$('light-label'),
        trigBtn=$('trigBtn'),
        vlm=$('vlm'), capImg=$('capImg'), capEmpty=$('capEmpty'),
        logEl=$('log'),
        replayBtn=$('replayBtn'), stopBtn=$('stopBtn'), ttsMsg=$('ttsMsg');

  let lastNarr=null;

  /* ---- SSE ---- */
  function connect(){
    const es=new EventSource('/events');
    es.onopen=()=>{sseDot.classList.add('ok');sseDot.title='SSE connected'};
    es.onmessage=e=>{
      try{handle(JSON.parse(e.data))}catch(err){console.error('SSE parse',err)}
    };
    es.onerror=()=>{
      sseDot.classList.remove('ok');sseDot.title='SSE disconnected';
      setTimeout(()=>{if(es.readyState===EventSource.CLOSED)connect()},2000);
    };
  }

  function handle(evt){
    switch(evt.type){
      case 'status': phase(evt.data.phase); break;
      case 'trigger':
        if(evt.data.active){light.classList.add('on');lightLabel.textContent='Triggered!'}
        else{light.classList.remove('on');lightLabel.textContent='Listening'}
        break;
      case 'vlm_text':
        vlm.textContent=evt.data.text; vlm.classList.remove('ph');
        lastNarr=evt.data.text; replayBtn.disabled=false;
        break;
      case 'captured_image':
        capImg.src=evt.data.url; capImg.style.display='block'; capEmpty.style.display='none';
        break;
      case 'error': addLog('error',evt.data.message); break;
      case 'log':   addLog(evt.data.level,evt.data.message); break;
    }
  }

  const phaseLabels={idle:'Idle',triggered:'Triggered',capturing:'Capturing Image...',
                     processing_vlm:'Processing (VLM)...',speaking:'Speaking...'};

  function phase(p){
    badge.textContent=phaseLabels[p]||p;
    badge.classList.toggle('active',p!=='idle');
    trigBtn.disabled=(p!=='idle');
    if(p!=='idle') lightLabel.textContent=phaseLabels[p]||p;
  }

  function addLog(level,msg){
    const d=document.createElement('div');
    d.className='le '+level; d.textContent=msg;
    logEl.appendChild(d); logEl.scrollTop=logEl.scrollHeight;
    while(logEl.children.length>300) logEl.removeChild(logEl.firstChild);
  }

  /* ---- Buttons ---- */
  trigBtn.onclick=async()=>{
    trigBtn.disabled=true;
    try{await fetch('/trigger',{method:'POST'})}
    catch(e){addLog('error','Trigger failed: '+e.message);trigBtn.disabled=false}
  };

  replayBtn.onclick=async()=>{
    if(!lastNarr)return;
    replayBtn.disabled=true; ttsMsg.textContent='Speaking...';
    try{
      const r=await fetch('/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:lastNarr})});
      const d=await r.json();
      ttsMsg.textContent=d.status==='ok'?'Done.':(d.error||'Failed');
    }catch(e){ttsMsg.textContent='Error: '+e.message}
    replayBtn.disabled=false;
  };

  stopBtn.onclick=async()=>{
    try{await fetch('/stop_tts',{method:'POST'});ttsMsg.textContent='Stopped.'}
    catch(e){ttsMsg.textContent='Error: '+e.message}
  };

  /* ---- Boot ---- */
  connect();
  addLog('info','Dev portal loaded. Waiting for events...');
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed_route():
    return video_feed()


@app.route("/events")
def events_route():
    """SSE stream — one long-lived connection per browser tab."""
    def gen():
        q: queue.Queue = queue.Queue(maxsize=200)
        with _sse_clients_lock:
            _sse_clients.append(q)
        try:
            while True:
                try:
                    payload = q.get(timeout=30)
                    yield f"data: {payload}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_clients_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(gen(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/trigger", methods=["POST"])
def trigger_route():
    """Manual pipeline trigger from the UI button."""
    broadcast_event("trigger", {"active": True})
    broadcast_event("status", {"phase": "triggered"})
    threading.Thread(target=_run_pipeline, args=("manual",), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/captured_image")
def captured_image_route():
    """Serve the most recently captured frame as JPEG."""
    with _captured_jpeg_lock:
        jpeg = _captured_jpeg
    if jpeg is None:
        return Response(status=204)
    return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.route("/capture", methods=["POST"])
def capture_route():
    """Fire the pipeline (async). Results arrive via SSE."""
    broadcast_event("trigger", {"active": True})
    broadcast_event("status", {"phase": "triggered"})
    threading.Thread(target=_run_pipeline, args=("web",), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/speak", methods=["POST"])
def speak_route():
    """Replay TTS for given text."""
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


@app.route("/stop_tts", methods=["POST"])
def stop_tts_route():
    """Kill the current aplay process to stop audio playback."""
    with _aplay_lock:
        proc = _aplay_process
    if proc is not None:
        try:
            proc.terminate()
            broadcast_event("status", {"phase": "idle"})
            logger.info("TTS playback stopped by user")
        except Exception:
            pass
    return jsonify({"status": "ok"})


@app.route("/tts_play", methods=["POST"])
def tts_play_route():
    """Synthesize and play arbitrary text (legacy endpoint)."""
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


# ---------------------------------------------------------------------------
# Voice trigger (Vosk)
# ---------------------------------------------------------------------------

_VOSK_RATE = 16000
_VOSK_CHANNELS = 1
_VOSK_CHUNK = 4000  # ~250 ms at 16 kHz
_VOSK_TRIGGER = "capture image"
_VOSK_COOLDOWN = 3.0
_VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-en-us-0.15")


def _find_emeet_device_index(pa) -> int | None:
    """Scan PyAudio input devices for the EMEET mic."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info.get("name", "")
        if info.get("maxInputChannels", 0) > 0 and ("EMEET" in name or "M0" in name):
            logger.info("Found EMEET mic: index=%d, name=%r", i, name)
            return i
    return None


def _generate_beep_wav(path: str, freq: int = 880, duration: float = 0.15):
    """Write a short sine-wave beep to a WAV file."""
    n_samples = int(_VOSK_RATE * duration)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_VOSK_RATE)
        for i in range(n_samples):
            sample = int(24000 * math.sin(2 * math.pi * freq * i / _VOSK_RATE))
            wf.writeframes(struct.pack("<h", sample))


def _play_beep():
    """Generate a beep WAV and play it through the speaker."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _generate_beep_wav(path)
        subprocess.run(["aplay", "-D", ALSA_DEVICE, path], capture_output=True, timeout=5)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _voice_listener_loop():
    """Background thread: listen for trigger phrase, then spawn pipeline thread."""
    try:
        import pyaudio
        import vosk
    except ImportError as e:
        logger.warning("Voice listener unavailable (missing dependency: %s)", e)
        return

    if not os.path.isdir(_VOSK_MODEL_PATH):
        logger.warning("Vosk model not found at %s — voice trigger disabled", _VOSK_MODEL_PATH)
        return

    vosk.SetLogLevel(-1)
    model = vosk.Model(_VOSK_MODEL_PATH)
    recognizer = vosk.KaldiRecognizer(model, _VOSK_RATE)

    pa = pyaudio.PyAudio()
    device_index = _find_emeet_device_index(pa)

    stream_kwargs = {
        "format": pyaudio.paInt16,
        "channels": _VOSK_CHANNELS,
        "rate": _VOSK_RATE,
        "input": True,
        "frames_per_buffer": _VOSK_CHUNK,
    }
    if device_index is not None:
        stream_kwargs["input_device_index"] = device_index
    else:
        logger.warning("EMEET mic not found, using default input device")

    try:
        stream = pa.open(**stream_kwargs)
    except Exception as e:
        logger.error("Could not open mic stream: %s", e)
        pa.terminate()
        return

    logger.info("Voice listener active — say '%s' to trigger capture", _VOSK_TRIGGER)

    last_trigger = 0.0
    try:
        while True:
            data = stream.read(_VOSK_CHUNK, exception_on_overflow=False)

            if _voice_busy.is_set():
                continue

            triggered = False
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if _VOSK_TRIGGER in text and (time.time() - last_trigger) > _VOSK_COOLDOWN:
                    triggered = True
                    logger.info("Voice trigger (final): %r", text)
            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "")
                if _VOSK_TRIGGER in text and (time.time() - last_trigger) > _VOSK_COOLDOWN:
                    triggered = True
                    logger.info("Voice trigger (partial): %r", text)

            if triggered:
                last_trigger = time.time()
                _voice_busy.set()
                broadcast_event("trigger", {"active": True})
                broadcast_event("status", {"phase": "triggered"})
                threading.Thread(target=_play_beep, daemon=True).start()
                threading.Thread(
                    target=_run_pipeline, args=("voice",), daemon=True,
                ).start()
                recognizer = vosk.KaldiRecognizer(model, _VOSK_RATE)
                last_trigger = time.time()
    except Exception as e:
        logger.error("Voice listener crashed: %s", e, exc_info=True)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        logger.info("Voice listener stopped")


def _start_voice_listener():
    """Start the voice trigger listener as a background daemon thread."""
    t = threading.Thread(target=_voice_listener_loop, daemon=True, name="vosk-listener")
    t.start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_camera()
    if not _camera_ready:
        logger.warning("Camera not available - stream will be blank")
    _start_voice_listener()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
