# Import libraries
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt

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

print("TF-IDF...")
# Prepare TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X = tfidf.fit_transform(lyrics)
y = data['emotion_encoded']

# Train-test split (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Logistic Regression...")
# Train a Logistic Regression classifier
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

print("Predictions...")
# Make predictions on the validation set
y_pred = lr_model.predict(X_val)

print("Evaluating the model...")
# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"Validation Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=emotion_classes))

# Save the TF-IDF vectorizer and Logistic Regression model
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(lr_model, 'logistic_regression_emotion_model.pkl')

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
