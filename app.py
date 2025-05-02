from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("belle_voix_model.h5")

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
FIXED_TIME_STEPS = 130

def preprocess_audio(file, max_len=SAMPLES_PER_TRACK):
    y, sr = librosa.load(file, sr=SAMPLE_RATE, duration=3)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    if mel_db.shape[1] < FIXED_TIME_STEPS:
        mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_TIME_STEPS - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :FIXED_TIME_STEPS]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db[np.newaxis, ..., np.newaxis]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    processed = preprocess_audio(file)
    prediction = model.predict(processed)[0][0]
    label = 'belle_voix' if prediction > 0.5 else 'non_belle_voix'
    return jsonify({'prediction': label, 'confidence': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
