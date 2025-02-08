# Import libraries
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
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

print("tf-idf....")
# Prepare TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X = tfidf.fit_transform(lyrics)
y = data['emotion_encoded']

# Train-test split (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("model...")
# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

print("prediction....")
# Make predictions on the validation set
y_pred = nb_model.predict(X_val)
y_proba = nb_model.predict_proba(X_val)

print("evaluation....")
# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

# Compute AUC-ROC score
# Binarize the labels for AUC-ROC calculation
y_val_binarized = label_binarize(y_val, classes=list(range(len(emotion_classes))))
auc_roc = roc_auc_score(y_val_binarized, y_proba, multi_class='ovr')

print(f"Validation Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC-AUC Score: {auc_roc}")
print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=emotion_classes))

# Save the TF-IDF vectorizer and Naive Bayes model
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(nb_model, 'naive_bayes_emotion_model.pkl')

print("recommendation...")
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
