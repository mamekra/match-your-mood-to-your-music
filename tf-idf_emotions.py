import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load the labeled dataset and TF-IDF matrix
dataset_path = 'C:/Users/user/Documents/ΠΜΣ DWS/NLP/project/labeled_songs.csv'
tfidf_matrix_path = 'C:/Users/user/Documents/ΠΜΣ DWS/NLP/project/tfidf_matrix.joblib'
vectorizer_path = 'C:/Users/user/Documents/ΠΜΣ DWS/NLP/project/tfidf_vectorizer.joblib'

df = pd.read_csv(dataset_path)
tfidf_matrix = joblib.load(tfidf_matrix_path)
vectorizer = joblib.load(vectorizer_path)


# Function to recommend songs based on emotion and TF-IDF similarity
def recommend_songs_by_emotion(input_song_title, top_k=5):
    """
    Recommends songs based on the emotion of the input song and TF-IDF similarity.

    Parameters:
    - input_song_title (str): The title of the song for which recommendations are needed.
    - top_k (int): The number of recommended songs to return.

    Returns:
    - A DataFrame with the recommended songs and their similarity scores.
    """
    # Check if the input song exists in the dataset
    if input_song_title not in df['title'].values:
        print("Song not found in the dataset.")
        return pd.DataFrame()

    # Get the index of the input song
    input_index = df[df['title'] == input_song_title].index[0]

    # Get the emotion of the input song
    input_emotion = df.loc[input_index, 'emotion'][0]  # Extract the emotion from the tuple

    # Compute cosine similarity between the input song and all other songs
    similarity_scores = cosine_similarity(tfidf_matrix[input_index], tfidf_matrix).flatten()

    # Create a DataFrame with titles and similarity scores
    recommendations = pd.DataFrame({'title': df['title'], 'similarity': similarity_scores, 'emotion': df['emotion']})

    # Filter songs by matching emotion
    recommendations = recommendations[recommendations['emotion'].apply(lambda x: x[0]) == input_emotion]

    # Sort recommendations by similarity in descending order
    recommendations = recommendations.sort_values(by='similarity', ascending=False)

    # Exclude the input song from the recommendations and return the top K results
    recommendations = recommendations[recommendations['title'] != input_song_title].head(top_k)

    return recommendations


# Example usage
input_song = "Can I Live"  # Replace with any song title from your dataset
top_k_recommendations = recommend_songs_by_emotion(input_song, top_k=5)

print(f"Top {len(top_k_recommendations)} recommendations for '{input_song}':")
print(top_k_recommendations)
