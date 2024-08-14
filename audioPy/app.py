from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import pygame
import io
import threading

app = Flask(__name__)
CORS(app)  # Allow all origins by default

def load_audio(file):
    y, sr = librosa.load(file)
    return y, sr

def detect_taps(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    taps = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    taps_time = librosa.frames_to_time(taps, sr=sr)
    return taps_time.tolist()

def map_to_beat(taps, beat_length=4):
    intervals = np.diff(taps)
    if len(intervals) == 0:
        return np.zeros(int(beat_length * 4)).tolist(), 0
    min_interval = np.min(intervals)
    quantization_grid = np.arange(0, taps[-1], min_interval)
    beat = np.zeros(len(quantization_grid))
    for tap in taps:
        closest_index = np.argmin(np.abs(quantization_grid - tap))
        beat[closest_index] = 1
    return beat.tolist(), min_interval

def create_drum_sounds():
    pygame.mixer.init()
    kick = pygame.mixer.Sound("808 mafia kick.wav")
    snare = pygame.mixer.Sound("Alex Snare.wav")
    hihat = pygame.mixer.Sound("Drip Hihat.wav")
    return [kick, snare, hihat]

def play_beat_loop(beat, drum_sounds, base_interval):
    beat_length = len(beat)
    while True:
        for i in range(beat_length):
            if beat[i] == 1:
                if i % 16 == 0:
                    drum_sounds[0].play()  # Kick
                elif i % 16 == 8:
                    drum_sounds[1].play()  # Snare
                else:
                    drum_sounds[2].play()  # Hi-hat
            pygame.time.wait(int(base_interval * 1000))

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save file to a temporary path
        temp_file_path = tempfile.mktemp(suffix='.wav')
        file.save(temp_file_path)

        # Load and process the audio file
        y, sr = load_audio(temp_file_path)
        taps = detect_taps(y, sr)
        beat, min_interval = map_to_beat(taps)

        # Clean up temporary file
        os.remove(temp_file_path)

        return jsonify({'taps': taps, 'beat': beat, 'min_interval': min_interval})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
