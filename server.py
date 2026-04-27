import numpy as np
import tensorflow as tf
import collections
import threading
import queue
import time
import sounddevice as sd
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# --- Конфігурація ---
MODEL_PATH     = 'audio_encoder_v3.tflite'
CENTROIDS_PATH = 'centroids.npy'
STDS_PATH      = 'stds.npy'

NORM_MEAN      = 0.7796
NORM_STD       = 0.1989
COMMANDS       = ['noise', 'marvin', 'go', 'stop', 'left', 'right', 'yes', 'no', 'unrecognized']
SAMPLE_RATE    = 16000
STEP_SAMPLES   = int(SAMPLE_RATE * 0.2) 
COOLDOWN_SEC   = 0.7 
MARGIN_THRESHOLD = 0.1 # Мінімальна різниця між 1-м та 2-м кандидатом

# --- Завантаження центроїдів ---
centroids_dict = np.load(CENTROIDS_PATH, allow_pickle=True).item()
stds_dict = np.load(STDS_PATH, allow_pickle=True).item()

centroid_matrix = np.array([centroids_dict[i] for i in range(9)])
stds_vector = np.array([stds_dict[i] for i in range(9)])

# Розрахунок жорсткого ліміту (Hard Limit)
# Можна взяти як центроїд + 3.5 сигми або заздалегідь розрахований макс.
max_allowed_dists = stds_vector * 2.5

# --- Ініціалізація TFLite ---
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

def get_embedding(spec):
    spec = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], spec)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

class ListenerState:
    def __init__(self):
        self.is_listening = False
        self.audio_queue  = queue.Queue()
        self.audio_buffer = collections.deque(maxlen=SAMPLE_RATE)
        self.last_trigger = 0.0
        self.stream       = None
        self.thread       = None
        self.subscribers  = []
        self._loop        = None

    def set_loop(self, loop): self._loop = loop

    def broadcast(self, event: dict):
        if self._loop is None: return
        data = json.dumps(event, ensure_ascii=False)
        for q in self.subscribers:
            self._loop.call_soon_threadsafe(q.put_nowait, data)

state = ListenerState()

def audio_callback(indata, frames, time_info, status):
    state.audio_queue.put(indata[:, 0].copy())

def predict_loop():
    samples_since_last_step = 0
    state.audio_buffer.clear()

    while state.is_listening:
        try:
            chunk = state.audio_queue.get(timeout=0.5)
        except queue.Empty: continue

        state.audio_buffer.extend(chunk)
        samples_since_last_step += len(chunk)

        if samples_since_last_step < STEP_SAMPLES: continue
        samples_since_last_step = 0
        
        if len(state.audio_buffer) < SAMPLE_RATE: continue

        # Аналіз поточного вікна (без усереднення векторів)
        window = np.array(state.audio_buffer, dtype=np.float32)
        spec = (audio_to_melspec(window) - NORM_MEAN) / (NORM_STD + 1e-7)
        emb = get_embedding(spec)

        # 1. Рахуємо відстані до всіх центроїдів
        dists = np.linalg.norm(centroid_matrix - emb, axis=1)
        
        # Знаходимо два найближчих класи для перевірки Margin
        sorted_indices = np.argsort(dists)
        best_idx = int(sorted_indices[0])
        second_best_idx = int(sorted_indices[1])
        
        min_dist = dists[best_idx]
        margin = dists[second_best_idx] - min_dist
        
        now = time.time()

        # 2. Жорстка фільтрація
        # Клас 0 (noise) та 8 (unrecognized) не тригерять події
        if best_idx not in [0, 8]:
            # Перевірка на входження в радіус + Margin Rejection
            if min_dist <= max_allowed_dists[best_idx] and margin > MARGIN_THRESHOLD:
                
                if (now - state.last_trigger) > COOLDOWN_SEC:
                    label = COMMANDS[best_idx]
                    state.last_trigger = now
                    
                    print(f"[{time.strftime('%H:%M:%S')}] 🎯 TRIGGER: {label.upper()} (dist: {min_dist:.3f}, margin: {margin:.3f})")
                    
                    state.broadcast({
                        "type": "keyword",
                        "keyword": label,
                        "distance": round(float(min_dist), 4),
                        "margin": round(float(margin), 4)
                    })

# --- FastAPI Setup ---
app = FastAPI()

@app.on_event("startup")
async def startup(): state.set_loop(asyncio.get_event_loop())

@app.post("/start")
def start():
    if state.is_listening: return {"status": "ok"}
    state.is_listening = True
    state.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=STEP_SAMPLES, callback=audio_callback)
    state.stream.start()
    state.thread = threading.Thread(target=predict_loop, daemon=True)
    state.thread.start()
    return {"status": "started"}

@app.post("/stop")
def stop():
    state.is_listening = False
    if state.stream: state.stream.stop(); state.stream.close()
    return {"status": "stopped"}

@app.get("/events")
async def sse_events():
    q = asyncio.Queue(); state.subscribers.append(q)
    async def gen():
        try:
            yield "data: {\"type\": \"connected\"}\n\n"
            while True: yield f"data: {await q.get()}\n\n"
        finally: state.subscribers.remove(q)
    return StreamingResponse(gen(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)