import pandas as pd
from nrclex import NRCLex
import nltk
import joblib

nltk.download('punkt')

# Load a small portion of the dataset for testing
df = pd.read_csv('cleaned_songs.csv', nrows=100000)

# Function to label emotions using NRC Emotion Lexicon
def label_emotions(text):
    emotion = NRCLex(text)
    emotion_scores = emotion.raw_emotion_scores
    if emotion_scores:
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        return top_emotion
    else:
        return 'neutral'

# Apply the function to the lyrics column and reassign the DataFrame
df = df.assign(emotion=df['cleaned_lyrics'].apply(label_emotions))
print('label emotions successful')

# Check the DataFrame structure
print(df.info())
print(df[['cleaned_lyrics', 'emotion']].head())

# Save the DataFrame using Joblib
joblib.dump(df, 'labeled_songs.joblib')
print('DataFrame saved using Joblib')
