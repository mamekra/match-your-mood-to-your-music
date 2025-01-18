# Import libraries
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import spacy

# Load spaCy's pre-trained English model for embeddings
nlp = spacy.load("en_core_web_md")

# Load the dataset
DATA_PATH = 'C:/Users/user/Documents/ΠΜΣ DWS/NLP/project/labeled_songs.csv'
data = pd.read_csv(DATA_PATH)

# Extract cleaned lyrics and emotion labels
lyrics = data['cleaned_lyrics']
emotions = data['emotion']

# Encode emotions as integers
emotion_classes = sorted(emotions.unique())
emotion_to_int = {emotion: i for i, emotion in enumerate(emotion_classes)}
data['emotion_encoded'] = data['emotion'].map(emotion_to_int)

# Function to compute average embeddings for lyrics
def get_embedding(text):
    doc = nlp(text)
    return np.mean([token.vector for token in doc if token.has_vector], axis=0)

# Compute embeddings for all lyrics
print("Computing embeddings...")
X_embeddings = np.array([get_embedding(lyric) for lyric in lyrics])
y = data['emotion_encoded']

# Train-test split (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

# Train an SVM classifier
print("Training SVM...")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the validation set
print("Making predictions...")
y_pred = svm_model.predict(X_val)
y_pred_proba = svm_model.predict_proba(X_val)

# Evaluate the model
print("Evaluating the model...")
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=emotion_classes))

# Compute ROC-AUC score
roc_auc = roc_auc_score(pd.get_dummies(y_val), y_pred_proba, multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc}")

# Save the embedding-based SVM model
joblib.dump(svm_model, 'svm_emotion_model_embeddings.pkl')

# Function to recommend a song based on emotion
def recommend_song(emotion, data, emotion_to_int):
    """
    Recommend a song based on the given emotion.
    """
    emotion_index = emotion_to_int.get(emotion)
    if emotion_index is None:
        return f"Emotion '{emotion}' not recognized."

    # Filter songs with the given emotion
    matching_songs = data[data['emotion_encoded'] == emotion_index]
    if matching_songs.empty:
        return f"No songs found for emotion '{emotion}'."

    # Randomly pick a matching song
    song = matching_songs.sample(1).iloc[0]
    return f"Recommended song: {song['title']} by {song['artist']}"

# Test the recommendation function
print(recommend_song('anger', data, emotion_to_int))
