import os
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import threading
import time
import queue
from collections import deque
from datetime import datetime

# ------------------- CONFIGURATION --------------------
SAMPLE_RATE = 16000            # Sample rate for audio (Hz)
CHANNELS = 1                   # Mono audio
AUDIO_BLOCK_SIZE = 1024        # Audio block size for streaming
AUDIO_BUFFER_SECONDS = 10      # Keep 10 seconds of audio before honk
VIDEO_FPS = 20                 # Webcam frames per second
VIDEO_BUFFER_SECONDS = 10      # Keep 10 seconds of video before honk
VIDEO_WIDTH = 640              # Webcam frame width
VIDEO_HEIGHT = 480             # Webcam frame height
DEBUG = True                   # Enable debug mode for predictions
CONFIDENCE_THRESHOLD = 0.1     # Confidence to detect "horn"
COOLDOWN_SECONDS = 5           # Prevent multiple triggers in short time
# ------------------------------------------------------

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load YAMNet model and class labels
print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]
print("✅ YAMNet loaded.")

# Buffers for audio and video data
audio_queue = queue.Queue()
audio_buffer = deque(maxlen=SAMPLE_RATE * AUDIO_BUFFER_SECONDS)
video_buffer = deque(maxlen=VIDEO_FPS * VIDEO_BUFFER_SECONDS)

# Stop signal for clean exit
stop_event = threading.Event()

# Audio input callback to stream data into buffer
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️ Audio Status:", status)
    audio_queue.put(indata.copy())

# Run YAMNet and detect if a horn-like sound is present
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

# Save buffered audio to WAV file
def save_audio(buffer):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_audio_{timestamp}.wav"
    data = np.concatenate(buffer, axis=0)
    sf.write(filename, data, SAMPLE_RATE)
    print(f"✅ Saved AUDIO: {filename}")

# Save buffered video to AVI file
def save_video(frames):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_video_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"✅ Saved VIDEO: {filename}")

# Thread to process audio and trigger detection
def process_audio():
    last_detection_time = 0

    while not stop_event.is_set():
        try:
            # Get audio block and add to buffer
            block = audio_queue.get(timeout=1)
            audio_buffer.append(block)

            # Only run detection when buffer has enough data
            if len(audio_buffer) * AUDIO_BLOCK_SIZE >= SAMPLE_RATE:
                last_second = np.concatenate(list(audio_buffer)[-SAMPLE_RATE:], axis=0).flatten()
                now = time.time()

                # Check cooldown before detecting again
                if now - last_detection_time > COOLDOWN_SECONDS:
                    if detect_honk(last_second):
                        print("🚗 Car horn detected!")

                        # Save pre-event buffers
                        save_audio(list(audio_buffer))
                        save_video(list(video_buffer))

                        # Clear buffers to avoid repeated detections
                        audio_buffer.clear()
                        video_buffer.clear()

                        last_detection_time = now
                        time.sleep(1)  # optional pause
        except queue.Empty:
            continue

    print("🎧 Audio processing stopped.")

# Thread to continuously record video to a buffer
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

# Main function to run both threads and manage shutdown
def main():
    print("🔍 Honk Detector Initialized (Press Ctrl+C to stop)")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=AUDIO_BLOCK_SIZE,
        callback=audio_callback
    )

    try:
        with stream:
            # Start background threads
            audio_thread = threading.Thread(target=process_audio)
            video_thread = threading.Thread(target=record_video)
            audio_thread.start()
            video_thread.start()

            # Main loop: wait until Ctrl+C
            while not stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected. Stopping...")
        stop_event.set()

    # Wait for threads to shut down cleanly
    print("🔄 Waiting for threads to finish...")
    audio_thread.join()
    video_thread.join()
    print("✅ Shutdown complete.")

# Entry point
if __name__ == "__main__":
    main()
