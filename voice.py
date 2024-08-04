import os
import numpy as np
import librosa
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pydub import AudioSegment

# Define phonemes and the directory
phonemes = ['a', 'e', 'i', 'o', 'u', 'b', 'c', 'd', 'f', 'g']
synthetic_sounds_dir = 'synthetic_sounds'

# Step 2: Extract Features
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

X = []
y = []

for phoneme in phonemes:
    file_path = os.path.join(synthetic_sounds_dir, f'{phoneme}.wav')
    mfccs = extract_mfcc(file_path)
    X.append(mfccs)
    y.append(phoneme)

X = np.array(X)
y = np.array(y)

# Step 3: Train the Model
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, y_encoded)

# Step 4: Evaluate and Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

def predict_phoneme(file_path):
    mfccs = extract_mfcc(file_path)
    mfccs = mfccs.reshape(1, -1)
    y_pred = model.predict(mfccs)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    return y_pred_decoded[0]

# Function to create a sentence from phoneme audio files
def create_sentence(phoneme_sequence, output_file='sentence.wav'):
    sentence_audio = AudioSegment.silent(duration=0)
    for phoneme in phoneme_sequence:
        phoneme_file = os.path.join(synthetic_sounds_dir, f'{phoneme}.wav')
        phoneme_audio = AudioSegment.from_wav(phoneme_file)
        sentence_audio += phoneme_audio + AudioSegment.silent(duration=100)  # Add a slight pause between phonemes
    sentence_audio.export(output_file, format="wav")

# Create a sentence
sentence_phonemes = ['b', 'a', 'd', 'e', 'i', 'o', 'u']  # Example phoneme sequence
create_sentence(sentence_phonemes)

# Function to predict phonemes from the generated sentence
def predict_phonemes_from_sentence(sentence_file, phoneme_sequence):
    predicted_phonemes = []

    # In real applications, you should segment the sentence file into phoneme segments
    # Here, we assume phoneme segments are given as phoneme_sequence for simplicity
    for phoneme in phoneme_sequence:
        phoneme_file = os.path.join(synthetic_sounds_dir, f'{phoneme}.wav')
        predicted_phoneme = predict_phoneme(phoneme_file)
        predicted_phonemes.append(predicted_phoneme)

    return predicted_phonemes

# Predict phonemes from the generated sentence
sentence_file = 'sentence.wav'
predicted_phonemes = predict_phonemes_from_sentence(sentence_file, sentence_phonemes)
print(f'Predicted phonemes: {predicted_phonemes}')
