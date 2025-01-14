import pandas as pd
from nrclex import NRCLex
import nltk
import joblib

nltk.download('punkt')

# Load a small portion of the dataset for testing
df = pd.read_csv('cleaned_songs.csv', nrows=100000)
OUTPUT_PATH = 'labeled_songs.csv'

# Function to label emotions using NRC Emotion Lexicon and return top emotion and its score
def label_emotions(text):
    """
    This function uses NRCLex to extract emotions and their scores.
    Returns the top emotion and its normalized score.
    """
    emotion = NRCLex(text)
    emotion_scores = emotion.raw_emotion_scores

    if emotion_scores:
        total_score = sum(emotion_scores.values())  # Sum of all emotion scores
        # Get the emotion with the highest score
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        top_score = emotion_scores[top_emotion] / total_score  # Normalize the top score
        return (top_emotion, round(top_score, 4))  # Return top emotion and rounded score
    else:
        return ('neutral', 1.0)  # Return 'neutral' with 100% probability if no emotion is detected

# Apply the function to the 'cleaned_lyrics' column
df['emotion'] = df['cleaned_lyrics'].apply(label_emotions)

print("Labeling top emotions and scores successful")
print(df[['cleaned_lyrics', 'emotion']].head())

# Save the DataFrame to a CSV file
df.to_csv(OUTPUT_PATH, index=False)
print(f"DataFrame saved to {OUTPUT_PATH}")

# Save the DataFrame using Joblib
joblib.dump(df, 'labeled_songs.joblib')
print('DataFrame saved using Joblib')
