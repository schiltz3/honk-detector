import cv2
import numpy as np
import sounddevice as sd
import queue
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
import subprocess
import os
import time
from datetime import datetime
from collections import deque
import threading
import signal
import sys

# ======================== CONFIGURATION ========================

# Audio parameters
SAMPLE_RATE = 16000
AUDIO_BLOCK_SIZE = 1024
AUDIO_BUFFER_SECONDS = 7
POST_EVENT_SECONDS = 7

# Video parameters
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 20
VIDEO_BUFFER_SECONDS = 7

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.08

# Cooldown time between detections (seconds)
COOLDOWN_SECONDS = 5

# Output directory
OUTPUT_DIR = "honk_clips"

# ======================== BUFFERS & FLAGS ========================

audio_queue = queue.Queue()
audio_buffer = deque(maxlen=(AUDIO_BUFFER_SECONDS + POST_EVENT_SECONDS) * SAMPLE_RATE // AUDIO_BLOCK_SIZE)
video_buffer = deque(maxlen=VIDEO_FPS * (VIDEO_BUFFER_SECONDS + POST_EVENT_SECONDS))

stop_event = threading.Event()

# ======================== AUDIO CALLBACK ========================

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    mono = np.mean(indata, axis=1)
    audio_queue.put(mono.copy())

# ======================== YAMNet SETUP ========================

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = [line.strip() for line in tf.io.gfile.GFile(class_map_path)]

def detect_honk(audio_data):
    horn_keywords = ["car horn", "vehicle horn", "truck horn", "air horn", "horn", "honking", "toot"]

    waveform = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)

    mean_scores = tf.reduce_mean(scores, axis=0)
    top_indices = tf.argsort(mean_scores, direction='DESCENDING')[:5]

    print("\n🔊 Top Predictions:")
    for i in top_indices:
        label_raw = class_names[i.numpy()]
        label = label_raw.split(',')[-1].strip()
        score = mean_scores[i].numpy()

        if any(keyword in label.lower() for keyword in horn_keywords):
            print(f"\033[92m- {label}: {score:.3f}\033[0m")
        else:
            print(f"- {label}: {score:.3f}")

    for i in top_indices:
        label_raw = class_names[i.numpy()]
        label = label_raw.split(',')[-1].strip().lower()
        score = mean_scores[i].numpy()
        if any(keyword in label for keyword in horn_keywords) and score > CONFIDENCE_THRESHOLD:
            return True
    return False

# ======================== SAVE AUDIO + VIDEO ========================

def save_audio_video(audio_buf, video_buf):
    if not video_buf or not audio_buf:
        print("⚠️ Skipping save: empty audio or video buffer.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wav_file = os.path.join(OUTPUT_DIR, f"temp_audio_{timestamp}.wav")
    raw_video_file = os.path.join(OUTPUT_DIR, f"temp_video_{timestamp}.avi")
    output_file = os.path.join(OUTPUT_DIR, f"honk_clip_{timestamp}.mp4")

    # Save audio
    audio_data = np.concatenate(audio_buf, axis=0)

    # Trim audio to match video duration
    expected_audio_samples = len(video_buf) * SAMPLE_RATE // VIDEO_FPS
    if len(audio_data) > expected_audio_samples:
        audio_data = audio_data[-expected_audio_samples:]
    elif len(audio_data) < expected_audio_samples:
        pad = expected_audio_samples - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad))

    sf.write(wav_file, audio_data, SAMPLE_RATE)
    print(f"✅ Saved AUDIO: {wav_file}")


    # Combine with FFmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', raw_video_file,
        '-i', wav_file,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-shortest',
        output_file
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🎬 Combined AUDIO + VIDEO saved as: {output_file}")
    except subprocess.CalledProcessError:
        print("❌ FFmpeg failed to combine audio and video.")

    # Optional cleanup
    # os.remove(wav_file)
    # os.remove(raw_video_file)

# ======================== VIDEO RECORDING ========================

def record_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            video_buffer.append(frame)
        time.sleep(1.0 / VIDEO_FPS)

    cap.release()
    print("🎥 Video capture stopped.")

# ======================== AUDIO PROCESSING ========================

def process_audio():
    last_detection_time = 0
    recording_in_progress = False

    while not stop_event.is_set():
        try:
            block = audio_queue.get(timeout=1)
            audio_buffer.append(block)

            if len(audio_buffer) * AUDIO_BLOCK_SIZE >= SAMPLE_RATE:
                recent_chunk = np.concatenate(list(audio_buffer)[-SAMPLE_RATE:], axis=0).flatten()
                now = time.time()

                # Skip if still in cooldown or already saving
                if (now - last_detection_time > COOLDOWN_SECONDS) and not recording_in_progress:
                    if detect_honk(recent_chunk):
                        print("🚗 Car horn detected!")
                        recording_in_progress = True

                        # Collect audio and video during post-event period
                        post_end = time.time() + POST_EVENT_SECONDS
                        while time.time() < post_end and not stop_event.is_set():
                            try:
                                post_block = audio_queue.get(timeout=0.1)
                                audio_buffer.append(post_block)
                            except queue.Empty:
                                pass

                            if video_buffer:
                                video_buffer.append(video_buffer[-1])  # repeat last frame if needed
                            time.sleep(1.0 / VIDEO_FPS)


                        # Copy current buffers
                        full_audio = list(audio_buffer).copy()
                        full_video = list(video_buffer).copy()

                        save_audio_video(full_audio, full_video)

                        # Clear for next event
                        audio_buffer.clear()
                        video_buffer.clear()

                        # Only now start cooldown
                        last_detection_time = time.time()
                        recording_in_progress = False

        except queue.Empty:
            continue

    print("🎧 Audio processing stopped.")


# ======================== CLEAN EXIT ========================

def signal_handler(sig, frame):
    print("🛑 Exiting program...")
    stop_event.set()

signal.signal(signal.SIGINT, signal_handler)

# ======================== MAIN ========================

if __name__ == "__main__":
    print("📡 Starting honk detection...")
    threading.Thread(target=record_video, daemon=True).start()

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=AUDIO_BLOCK_SIZE, callback=audio_callback):
        process_audio()

    print("✅ Program exited.")
