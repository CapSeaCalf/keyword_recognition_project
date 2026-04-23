import numpy as np
import tensorflow as tf
import collections
import threading
import queue
import time
import sounddevice as sd

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

# --- Конфігурація ---
MODEL_PATH   = 'keyword_v2.tflite'
NORM_MEAN    = 0.7796
NORM_STD     = 0.1989
COMMANDS     = ['noise', 'marvin', 'go', 'stop', 'left', 'right', 'yes', 'no']
SAMPLE_RATE  = 16000
STEP_SAMPLES = int(SAMPLE_RATE * 0.2)
VOTE_WINDOW  = 5
COOLDOWN_SEC = 1.0

confidence_thresholds = np.array([0.8, 0.9, 0.75, 0.92, 0.92, 0.85, 0.9, 0.75], dtype=np.float32)

# --- Модель ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def audio_to_melspec(audio):
    t = tf.constant(audio, dtype=tf.float32)
    stfts = tf.signal.stft(t, frame_length=512, frame_step=160, fft_length=512)
    spec  = tf.abs(stfts)
    mel_w = tf.signal.linear_to_mel_weight_matrix(64, spec.shape[-1], SAMPLE_RATE, 80.0, 7600.0)
    mel   = tf.tensordot(spec, mel_w, 1)
    mel_db   = tf.math.log(mel + 1e-6)
    mel_norm = tf.clip_by_value((mel_db + 13.8) / 13.8, 0, 1)
    mel_norm = tf.image.resize(mel_norm[tf.newaxis, :, :, tf.newaxis], [64, 128])[0, :, :, 0]
    return mel_norm.numpy()

def predict(spec):
    spec = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], spec)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# --- Стан програми ---
class ListenerState:
    def __init__(self):
        self.is_listening = False
        self.audio_queue  = queue.Queue()
        self.audio_buffer = collections.deque(maxlen=SAMPLE_RATE)
        self.vote_buffer  = collections.deque(maxlen=VOTE_WINDOW)
        self.last_trigger = 0.0
        self.stream       = None
        self.thread       = None
        # SSE subscribers: list of asyncio.Queue
        self.subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop):
        self._loop = loop

    def broadcast(self, event: dict):
        """Push event to all SSE subscribers (thread-safe)."""
        if self._loop is None:
            return
        data = json.dumps(event, ensure_ascii=False)
        for q in self.subscribers:
            self._loop.call_soon_threadsafe(q.put_nowait, data)

state = ListenerState()

# --- Аудіо-колбек ---
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    state.audio_queue.put(indata[:, 0].copy())

# --- Цикл передбачення ---
def predict_loop():
    samples_since_last_step = 0
    state.audio_buffer.clear()
    state.vote_buffer.clear()
    state.last_trigger = 0.0

    while state.is_listening:
        try:
            chunk = state.audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        state.audio_buffer.extend(chunk)
        samples_since_last_step += len(chunk)

        if samples_since_last_step < STEP_SAMPLES:
            continue
        samples_since_last_step = 0

        if len(state.audio_buffer) < SAMPLE_RATE:
            continue

        window   = np.array(state.audio_buffer, dtype=np.float32)
        spec     = audio_to_melspec(window)
        spec     = (spec - NORM_MEAN) / (NORM_STD + 1e-7)
        raw_pred = predict(spec)
        state.vote_buffer.append(raw_pred)

        if len(state.vote_buffer) == VOTE_WINDOW:
            mean_probs = np.mean(state.vote_buffer, axis=0)
            pred_class = int(np.argmax(mean_probs))
            conf       = mean_probs[pred_class] / np.linalg.norm(mean_probs)
            now        = time.time()

            if conf >= confidence_thresholds[pred_class] and pred_class != 0:
                if (now - state.last_trigger) > COOLDOWN_SEC:
                    label = COMMANDS[pred_class]
                    ts    = time.strftime('%H:%M:%S')
                    print(f"[{ts}] >>> Впізнано: {label} ({conf:.0%})")
                    state.last_trigger = now
                    state.vote_buffer.clear()
                    state.broadcast({
                        "type": "keyword",
                        "keyword": label,
                        "confidence": round(float(conf), 4),
                        "timestamp": ts,
                    })

# --- FastAPI ---
app = FastAPI(title="Keyword Spotting API")

@app.on_event("startup")
async def startup():
    state.set_loop(asyncio.get_event_loop())

@app.post("/start")
def start_listening():
    if state.is_listening:
        return {"status": "already_listening"}

    state.is_listening = True
    # Drain stale audio
    while not state.audio_queue.empty():
        state.audio_queue.get_nowait()

    state.stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=STEP_SAMPLES,
        callback=audio_callback,
    )
    state.stream.start()

    state.thread = threading.Thread(target=predict_loop, daemon=True)
    state.thread.start()

    print("Listening started.")
    return {"status": "started"}

@app.post("/stop")
def stop_listening():
    if not state.is_listening:
        return {"status": "not_listening"}

    state.is_listening = False

    if state.stream:
        state.stream.stop()
        state.stream.close()
        state.stream = None

    if state.thread:
        state.thread.join(timeout=2)
        state.thread = None

    print("Listening stopped.")
    return {"status": "stopped"}

@app.get("/status")
def get_status():
    return {"is_listening": state.is_listening}

@app.get("/events")
async def sse_events():
    """Server-Sent Events stream — pushes keyword detections to the client."""
    q: asyncio.Queue = asyncio.Queue()
    state.subscribers.append(q)

    async def event_generator():
        try:
            # Send an initial ping so the client knows we're connected
            yield "data: {\"type\": \"connected\"}\n\n"
            while True:
                data = await q.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            state.subscribers.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- Точка входу ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)