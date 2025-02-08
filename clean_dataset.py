import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Path to the dataset file
DATASET_PATH = 'song_lyrics.csv'  # Replace with the actual path to your dataset
OUTPUT_PATH = 'cleaned_songs.csv'  # Path to save the cleaned dataset

def clean_lyrics(text):
    """
    Function to clean song lyrics by removing unwanted characters and normalizing text.
    """
    # Remove metadata like [Chorus], [Verse]
    text = re.sub(r'\[.*?\]', '', text)
    # Remove special characters, digits, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text with explicit language set to 'english'
    tokens = word_tokenize(text, language='english')
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return ' '.join(tokens)

def main():
    """
    Main function to load, clean, and save the dataset.
    """
    print("Loading dataset...")
    # Load the dataset
    try:
        # Load only the first 100,000 rows from the Genius
        dataset = pd.read_csv(DATASET_PATH, nrows=100000)
    except FileNotFoundError:
        print(f"Error: File not found at {DATASET_PATH}. Please check the path and try again.")
        return

    # Keep relevant columns
    columns_to_keep = ['title', 'tag', 'artist', 'year', 'lyrics', 'language']
    dataset = dataset[columns_to_keep]

    # Filter for English songs
    print("Filtering for English songs...")
    english_songs = dataset[dataset['language'] == 'en']

    # Drop rows with missing lyrics and remove duplicates
    print("Cleaning dataset...")
    english_songs = english_songs.dropna(subset=['lyrics'])
    english_songs = english_songs.drop_duplicates(subset=['lyrics'])

    # Clean the lyrics column
    print("Processing lyrics...")
    english_songs['cleaned_lyrics'] = english_songs['lyrics'].apply(clean_lyrics)

    # Save the cleaned dataset
    print(f"Saving cleaned dataset to {OUTPUT_PATH}...")
    english_songs.to_csv(OUTPUT_PATH, index=False)
    print("Data cleaning complete!")

if __name__ == "__main__":
    main()