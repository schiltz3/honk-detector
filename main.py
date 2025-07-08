import os
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
DEBUG = True
CONFIDENCE_THRESHOLD = 0.1  # Lower threshold for debug
# ------------------------------------------------

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load YAMNet model and labels
print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]
print("YAMNet loaded.")

# Buffers and queues
audio_queue = queue.Queue()
audio_buffer = deque(maxlen=SAMPLE_RATE * AUDIO_BUFFER_SECONDS)  # stores raw audio blocks
video_buffer = deque(maxlen=VIDEO_BUFFER_SECONDS * VIDEO_FPS)    # stores video frames

# Event to signal threads to stop
stop_event = threading.Event()

# Audio callback function to enqueue recorded audio blocks
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio Status:", status)
    audio_queue.put(indata.copy())

# Function to run YAMNet detection on audio chunk
def detect_honk(audio_chunk):
    audio_chunk = audio_chunk.astype(np.float32)
    scores, _, _ = yamnet_model(audio_chunk)
    prediction = np.mean(scores.numpy(), axis=0)
    top_indices = np.argsort(prediction)[-5:][::-1]

    if DEBUG:
        print("\n🔊 Top Predictions:")
        for i in top_indices:
            print(f"{class_names[i]}: {prediction[i]:.3f}")

    for i in top_indices:
        label = class_names[i].lower()
        conf = prediction[i]
        if "horn" in label and conf > CONFIDENCE_THRESHOLD:
            return True
    return False

# Save audio buffer to WAV file
def save_audio(buffer):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_audio_{timestamp}.wav"
    data = np.concatenate(buffer, axis=0)
    sf.write(filename, data, SAMPLE_RATE)
    print(f"✅ Saved AUDIO: {filename}")

# Save video buffer to AVI file
def save_video(frames):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_video_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"✅ Saved VIDEO: {filename}")

# Thread function to process audio and detect honks
def process_audio():
    while not stop_event.is_set():
        try:
            block = audio_queue.get(timeout=1)
            audio_buffer.append(block)
            # Check if we have at least 1 second of audio for detection
            if len(audio_buffer) * AUDIO_BLOCK_SIZE >= SAMPLE_RATE:
                # Concatenate the last second of audio for detection
                last_second = np.concatenate(list(audio_buffer)[-int(SAMPLE_RATE / AUDIO_BLOCK_SIZE):], axis=0).flatten()
                if detect_honk(last_second):
                    print("🚗 Car horn detected!")
                    save_audio(list(audio_buffer))
                    save_video(list(video_buffer))
                    # Pause briefly to avoid multiple saves on same event
                    time.sleep(2)
        except queue.Empty:
            continue
    print("🎧 Audio processing stopped.")

# Thread function to capture video frames continuously
def record_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

    if not cap.isOpened():
        print("❌ Webcam not available.")
        stop_event.set()
        return

    print("📷 Capturing video...")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            video_buffer.append(frame)
        else:
            print("⚠️ Frame drop.")
        time.sleep(1.0 / VIDEO_FPS)

    cap.release()
    print("📷 Video capture stopped.")

# Main function to start audio stream and threads
def main():
    print("🔍 Honk Detector with Debug Mode ON")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=AUDIO_BLOCK_SIZE,
        callback=audio_callback
    )

    try:
        with stream:
            audio_thread = threading.Thread(target=process_audio)
            video_thread = threading.Thread(target=record_video)
            audio_thread.start()
            video_thread.start()

            print("🎧 Listening... Press Ctrl+C to stop.")
            while not stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected. Stopping...")
        stop_event.set()

    print("🔄 Waiting for threads to finish...")
    audio_thread.join()
    video_thread.join()
    print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
