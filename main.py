import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import cv2
import threading
import time
import queue
from collections import deque
from datetime import datetime

# ------------------- CONFIG --------------------
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_BLOCK_SIZE = 1024
AUDIO_BUFFER_SECONDS = 20
VIDEO_FPS = 20
VIDEO_BUFFER_SECONDS = 20
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.3
# ------------------------------------------------

# Load YAMNet
print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]
print("YAMNet loaded.")

# Audio buffer
audio_queue = queue.Queue()
audio_buffer = deque(maxlen=SAMPLE_RATE * AUDIO_BUFFER_SECONDS)

# Video buffer
video_buffer = deque(maxlen=VIDEO_BUFFER_SECONDS * VIDEO_FPS)

# Audio callback
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio Status:", status)
    audio_queue.put(indata.copy())

# Honk detection
def detect_honk(audio_chunk):
    audio_chunk = audio_chunk.astype(np.float32)
    scores, _, _ = yamnet_model(audio_chunk)
    prediction = np.mean(scores.numpy(), axis=0)
    top_indices = np.argsort(prediction)[-5:][::-1]
    print(f"Top class: {class_names[top_indices[0]]}, confidence: {prediction[top_indices[0]]:.3f}")

    for i in top_indices:
        label = class_names[i].lower()
        conf = prediction[i]
        print(f"Detected: {label} ({conf:.2f})")
        if "horn" in label and conf > CONFIDENCE_THRESHOLD:
            return True
    return False

# Save audio buffer
def save_audio(buffer):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"honk_audio_{timestamp}.wav"
    data = np.concatenate(buffer)
    sf.write(filename, data, SAMPLE_RATE)
    print(f"Audio saved: {filename}")

# Save video buffer
def save_video(frames):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"honk_video_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved: {filename}")

# Audio processor
def process_audio():
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)
        if len(audio_buffer) >= SAMPLE_RATE:
            last_second = np.concatenate(list(audio_buffer)[-SAMPLE_RATE:])
            if detect_honk(last_second):
                print("Honk detected!")
                save_audio(list(audio_buffer))
                save_video(list(video_buffer))
                time.sleep(3)  # Avoid repeated triggers

# Video capture thread
def record_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Recording video...")
    while True:
        ret, frame = cap.read()
        if ret:
            video_buffer.append(frame)
        else:
            print("Warning: Frame capture failed.")
        time.sleep(1.0 / VIDEO_FPS)

# Start everything
def main():
    print("Starting honk detection system...")
    stream = sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            blocksize=AUDIO_BLOCK_SIZE,
                            callback=audio_callback)

    with stream:
        audio_thread = threading.Thread(target=process_audio, daemon=True)
        video_thread = threading.Thread(target=record_video, daemon=True)
        audio_thread.start()
        video_thread.start()
        print("System is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            audio_thread.join()
            video_thread.join()
            exit()

if __name__ == "__main__":
    main()
