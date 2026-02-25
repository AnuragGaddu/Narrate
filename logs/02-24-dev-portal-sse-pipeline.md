# Dev Portal and SSE Pipeline Tracing — 2026-02-24

Complete rewrite of the Narrate web UI from a two-tab layout into a 4-column
developer portal with real-time pipeline tracing via Server-Sent Events (SSE).
The portal shows every stage of the capture-VLM-TTS pipeline as it executes.

---

## 1. Goal

Replace the tab-based UI (Narration Camera + Text to Speech) with a developer
portal that:

- Shows the pipeline status in real time (capturing, VLM processing, speaking).
- Displays the live camera stream and the captured image side by side.
- Shows VLM text output as it arrives.
- Provides a scrollable log panel with color-coded severity levels.
- Has trigger, replay, and stop controls.
- Updates entirely via server-push (SSE) rather than polling.

---

## 2. SSE Infrastructure

### 2.1 Server-side design

Each browser tab that connects to `GET /events` gets its own `queue.Queue`
(max size 200). All queues are tracked in a global list:

```python
_sse_clients: list[queue.Queue] = []
_sse_clients_lock = threading.Lock()
```

The `broadcast_event()` function pushes a JSON payload to every connected
client. Dead queues (full, meaning the browser is not consuming fast enough)
are removed:

```python
def broadcast_event(event_type: str, data=None):
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
```

### 2.2 SSE log handler

A custom `logging.Handler` forwards all Python `logger.*` calls to the
browser in real time:

```python
class _SSELogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        broadcast_event("log", {"level": record.levelname.lower(), "message": msg})
```

This means `logger.info("Pipeline started")` automatically appears in the
browser's log panel without any extra code at the call site.

### 2.3 `/events` endpoint

```python
@app.route("/events")
def events_route():
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
```

Keepalive comments (`: keepalive\n\n`) are sent every 30 seconds to prevent
proxies or browsers from closing idle connections.

### 2.4 Event types

| Event | Data | Sent when |
|-------|------|-----------|
| `status` | `{"phase": "idle\|triggered\|capturing\|processing_vlm\|speaking"}` | Pipeline stage changes |
| `trigger` | `{"active": true\|false}` | Voice or manual trigger starts/ends |
| `captured_image` | `{"url": "/captured_image?t=..."}` | Frame captured (cache-busted URL) |
| `vlm_text` | `{"text": "..."}` | VLM inference complete |
| `error` | `{"message": "..."}` | Any pipeline error |
| `log` | `{"level": "info\|warning\|error\|debug", "message": "..."}` | Every Python log record |

---

## 3. Pipeline Refactor

### 3.1 Before: blocking route handler

The old `/capture` route ran the entire pipeline synchronously — capture,
VLM inference (~12s), TTS synthesis, and speaker playback — all inside the
HTTP request handler. The browser had to wait for the full response.

### 3.2 After: async pipeline with SSE updates

The new `_run_pipeline(source)` function runs in a background daemon thread.
The HTTP route returns immediately with `{"status": "started"}`, and the
browser receives real-time updates via SSE:

```python
@app.route("/trigger", methods=["POST"])
def trigger_route():
    broadcast_event("trigger", {"active": True})
    broadcast_event("status", {"phase": "triggered"})
    threading.Thread(target=_run_pipeline, args=("manual",), daemon=True).start()
    return jsonify({"status": "started"})
```

The pipeline broadcasts its progress at each stage:

```
triggered -> capturing -> processing_vlm -> speaking -> idle
```

The voice listener was also updated to spawn `_run_pipeline("voice")` in a
thread rather than blocking the listener loop.

### 3.3 Captured image storage

`capture_frame()` was changed to return `(numpy_array, jpeg_bytes)` instead
of just the numpy array. The raw JPEG bytes are stored in `_captured_jpeg`
and served by a new endpoint:

```python
@app.route("/captured_image")
def captured_image_route():
    with _captured_jpeg_lock:
        jpeg = _captured_jpeg
    if jpeg is None:
        return Response(status=204)
    return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-cache"})
```

This avoids re-encoding the frame to JPEG — the bytes from the MJPEG stream
are used directly.

---

## 4. New Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/events` | SSE stream for real-time pipeline updates |
| `POST` | `/trigger` | Manual pipeline trigger from the UI button |
| `GET` | `/captured_image` | Serves the most recently captured frame as JPEG |
| `POST` | `/stop_tts` | Terminates the current `aplay` process |

---

## 5. TTS Stop Control

### 5.1 Problem

Previously, `_play_wav_on_speaker()` used `subprocess.run()` which blocked
until playback finished. There was no way to interrupt a long narration.

### 5.2 Fix

Changed to `subprocess.Popen()` with the process handle stored in a global:

```python
_aplay_process = None
_aplay_lock = threading.Lock()
```

The stop endpoint sends SIGTERM:

```python
@app.route("/stop_tts", methods=["POST"])
def stop_tts_route():
    with _aplay_lock:
        proc = _aplay_process
    if proc is not None:
        proc.terminate()
        broadcast_event("status", {"phase": "idle"})
    return jsonify({"status": "ok"})
```

In `_play_wav_on_speaker()`, return code -15 (SIGTERM) is treated as
intentional and not logged as an error.

---

## 6. UI Overhaul — 4-Column Dev Portal

### 6.1 Layout

```
+----------+----------------+----------------+----------+
| Trigger  | Video / Log    | Image / Caption| TTS      |
|          |                |                |          |
| [light]  | [live stream]  | [captured img] | [Replay] |
| [button] | [log panel]    | [VLM output]   | [Stop]   |
+----------+----------------+----------------+----------+
```

CSS grid: `grid-template-columns: 180px 1fr 1fr 160px`

Responsive: collapses to single column below 900px.

### 6.2 Column 1 — Trigger

- Status light: red circle (listening) / green circle (triggered/active).
- Label text updates with pipeline phase.
- Manual "Trigger" button to start the pipeline from the UI.

### 6.3 Column 2 — Video / Log

- Live MJPEG stream from `/video_feed`.
- Scrollable log panel showing all SSE `log` events, color-coded by level
  (info: gray, warning: yellow, error: red, debug: dark gray).
- Log capped at 300 entries to prevent memory growth.

### 6.4 Column 3 — Image / Caption

- Captured image (updated via SSE `captured_image` event).
- VLM text output (updated via SSE `vlm_text` event).

### 6.5 Column 4 — TTS

- "Replay" button: re-speaks the last narration via `POST /speak`.
- "Stop" button: kills `aplay` via `POST /stop_tts`.
- Status message area.

### 6.6 Header

- App title "Narrate".
- Pipeline badge showing current phase (e.g. "Processing (VLM)...").
- SSE connection dot (red = disconnected, green = connected).

### 6.7 JavaScript SSE client

```javascript
const es = new EventSource('/events');
es.onmessage = e => handle(JSON.parse(e.data));
```

Auto-reconnects on disconnect with a 2-second delay. The `handle()` function
dispatches events to update the status light, pipeline badge, captured image,
VLM text, log panel, and button states.

---

## 7. Architectural Decisions

### 7.1 SSE vs. WebSocket vs. polling

Three real-time update strategies were considered:

| Approach | Verdict | Reason |
|----------|---------|--------|
| **Server-Sent Events** | Chosen | One-direction server-to-client push is all that is needed. Native browser `EventSource` API with built-in auto-reconnect. No extra Python dependencies. Simple text protocol over HTTP. |
| WebSocket | Rejected | Bidirectional communication is unnecessary -- the browser only needs to receive events, not send them. Adds complexity (ws library, connection upgrade, ping/pong). |
| Polling | Rejected | Introduces latency (minimum one polling interval before updates arrive). Wastes bandwidth on empty responses. Would require client-side timer management. |

SSE was the simplest option that met the requirement: push pipeline status
to the browser in real time with sub-second latency.

### 7.2 Complete rewrite vs. incremental patching

The existing UI was a two-tab layout (Narration Camera + Text to Speech)
with scattered `disabled = true/false` button state management and
browser-side audio playback. The target was a 4-column grid with SSE-driven
updates, server-side audio, and a log panel.

The two UIs shared almost no HTML structure, CSS, or JavaScript logic.
Attempting to incrementally patch the old UI into the new one would have
required more careful surgery (and more potential for breakage) than simply
writing the new HTML/CSS/JS block from scratch while preserving the Python
backend functions.

The rewrite replaced the entire `HTML = """..."""` string in `app.py` and
the associated route handlers, but left the camera, VLM, TTS, and voice
listener code untouched.

---

## 8. Files Changed

| File | Change |
|------|--------|
| `app.py` | Added SSE infrastructure (`broadcast_event`, `_SSELogHandler`, `/events`). Replaced tab-based HTML with 4-column dev portal. Refactored pipeline into `_run_pipeline()` running in background thread. Added `/trigger`, `/captured_image`, `/stop_tts` endpoints. Changed `_play_wav_on_speaker()` to use `Popen` for stop support. Modified `capture_frame()` to return `(array, jpeg_bytes)` tuple. |
