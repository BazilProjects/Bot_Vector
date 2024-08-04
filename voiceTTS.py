import numpy as np
import librosa
import librosa.display
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pydub import AudioSegment
import sounddevice as sd
import os
# Define phonemes and create directory
phonemes = ['a', 'e', 'i', 'o', 'u', 'b', 'c', 'd', 'f', 'g']
synthetic_sounds_dir = 'synthetic_sounds'
# Define phonemes and create directory
phonemes = ['a', 'e', 'i', 'o', 'u', 'b', 'c', 'd', 'f', 'g']
synthetic_sounds_dir = 'synthetic_sounds'
os.makedirs(synthetic_sounds_dir, exist_ok=True)

# Function to extract MFCCs
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

X = []  # Phonemes as features (one-hot encoded)
y = []  # MFCCs as targets

for phoneme in phonemes:
    file_path = os.path.join(synthetic_sounds_dir, f'{phoneme}.wav')
    if os.path.exists(file_path):
        mfccs = extract_mfcc(file_path)
        X.append(phonemes.index(phoneme))  # One-hot encode the phoneme
        y.append(mfccs)

X = np.array(X).reshape(-1, 1)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = ExtraTreesRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# Function to extract MFCCs
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Display MFCCs
def display_mfccs(mfccs):
    plt.figure(figsize=(10, 4))
    mfccs_2d = mfccs.reshape(1, -1)  # Reshape to 2D array
    plt.imshow(mfccs_2d, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Predicted MFCCs')
    plt.colorbar()
    plt.show()

# Reconstruct audio from MFCCs (placeholder function)
def reconstruct_audio_from_mfccs(mfccs, sr=16000):
    # Placeholder: This is a very simple approximation. 
    # Real audio reconstruction would require a more complex method.
    y = librosa.feature.inverse.mfcc_to_audio(mfccs, sr=sr)
    return y

# Play audio
def play_audio(audio_data, sr=16000):
    sd.play(audio_data, samplerate=sr)
    sd.wait()

# Predict MFCCs for a given sentence
def predict_mfccs(text):
    mfccs_sequence = []
    for char in text:
        if char in phonemes:
            phoneme_index = phonemes.index(char)
            predicted_mfccs = model.predict(np.array([phoneme_index]).reshape(-1, 1))
            mfccs_sequence.append(predicted_mfccs.flatten())
    
    mfccs_sequence = np.array(mfccs_sequence)
    mean_mfccs = np.mean(mfccs_sequence, axis=0)
    display_mfccs(mean_mfccs)

    # Reconstruct and play audio
    audio_data = reconstruct_audio_from_mfccs(mean_mfccs)
    play_audio(audio_data)

# Example usage
sentence_text = 'iloveu'
predict_mfccs(sentence_text)
