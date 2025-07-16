import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from collections import Counter

# Path to audio folder
AUDIO_FOLDER = "audio"

# Function to extract MFCC features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)  # Load only 3 seconds
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)

# Load features and labels
features = []
labels = []

for file_name in os.listdir(AUDIO_FOLDER):
    if file_name.endswith(".wav"):
        file_path = os.path.join(AUDIO_FOLDER, file_name)
        data = extract_features(file_path)
        features.append(data)

        # Labeling
        if "not_depressed" in file_name.lower():
            labels.append(0)  # Not Depressed
        elif "depressed" in file_name.lower():
            labels.append(1)  # Depressed
        else:
            print(f"Skipping {file_name} (no valid label found)")

# Check label distribution
print("Label distribution:", Counter(labels))

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Ensure at least two classes are present
unique_labels = np.unique(y)
print("Classes used in training:", unique_labels)
if len(unique_labels) < 2:
    raise ValueError("❌ Error: Training requires at least 2 classes (0 and 1).")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("\n✅ Model Trained Successfully!")
print("🎯 Accuracy:", round(acc * 100, 2), "%")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'depression_model.pkl')
print("💾 Model saved as depression_model.pkl")
