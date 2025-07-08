import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load and resample a test sound (horn or otherwise)
filename = "car_horn_sample.wav"  # Use your test file
waveform, sr = librosa.load(filename, sr=16000)

scores, embeddings, spectrogram = yamnet_model(waveform)
prediction = np.mean(scores.numpy(), axis=0)

# Load labels
labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
class_names = [line.strip().split(',')[2] for line in tf.keras.utils.get_file('yamnet_class_map.csv', labels_url).readlines()[1:]]

top5 = np.argsort(prediction)[-5:][::-1]
for i in top5:
    print(f"{class_names[i]}: {prediction[i]:.3f}")
