import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

# Load the dataset
file_path = r'C:/Users/user/Documents/ΠΜΣ DWS/NLP/project/cleaned_songs.csv'
df = pd.read_csv(file_path)

# Ensure there are no missing values in the 'cleaned_lyrics' column
df = df.dropna(subset=['cleaned_lyrics'])

# Initialize the TF-IDF vectorizer and fit it on the 'cleaned_lyrics' column
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['cleaned_lyrics'])

# We do not convert to a dense array; keep it as a sparse matrix
document_vectors = tfidf_matrix


def preprocess_query(query, vectorizer):
    """Vectorizes the input query using the same TF-IDF vectorizer."""
    return vectorizer.transform([query])  # Return a sparse matrix


def get_k_similar_songs(input_query, k, metric='cosine'):
    """Finds the top K similar songs based on the given metric."""
    query_vector = vectorizer.transform([input_query])

    # Compute cosine similarity between the query vector and document vectors
    if metric == 'cosine':
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        top_k_indices = similarities.argsort()[-k:][::-1]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Retrieve the titles, lyrics, and similarities of the top K similar songs
    top_k_titles = [df.iloc[i]['title'] for i in top_k_indices]
    top_k_cleaned_lyrics = [df.iloc[i]['cleaned_lyrics'] for i in top_k_indices]
    top_k_similarities = similarities[top_k_indices]

    return top_k_titles, top_k_cleaned_lyrics, top_k_similarities


def extract_relevant_sentences(query, document, top_n=3):
    """Extracts the top N relevant sentences from a document based on the query."""
    sentences = sent_tokenize(document)
    sentence_vectors = vectorizer.transform(sentences)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, sentence_vectors).flatten()
    top_n_indices = similarities.argsort()[-top_n:][::-1]
    return [sentences[i] for i in top_n_indices]


def calculate_precision(query, query_vector, top_k_cleaned_lyrics):
    """Calculates precision based on cosine similarity with relevant sentences."""
    precision_scores = []
    for lyrics in top_k_cleaned_lyrics:
        relevant_sentences = extract_relevant_sentences(query, lyrics)
        sentence_vectors = vectorizer.transform(relevant_sentences)
        similarities = cosine_similarity(query_vector, sentence_vectors).flatten()
        precision_scores.append(np.mean(similarities))

    precision = np.mean(precision_scores)
    return precision


# Get input query from the user
input_query = input("Enter your query (word or sentence): ")

# Vectorize the input query
input_query_vector = preprocess_query(input_query, vectorizer)

# Ask the user for the number of similar songs to retrieve
k = int(input("How many similar songs do you want as output? (e.g., 5, 10): "))

# Retrieve the top K similar songs
top_k_titles, top_k_cleaned_lyrics, top_k_similarities = get_k_similar_songs(input_query, k)

# Display the top K similar songs
print(f"\nTop {k} similar songs for query '{input_query}':")
for i, (title, similarity) in enumerate(zip(top_k_titles, top_k_similarities)):
    print(f"\nSong {i + 1}: {title}")
    print(f"Similarity: {similarity:.4f}")
    relevant_sentences = extract_relevant_sentences(input_query, top_k_cleaned_lyrics[i])
    print("Relevant sentences:")
    for sentence in relevant_sentences:
        print(f"- {sentence}")

# Calculate and display precision
precision = calculate_precision(input_query, input_query_vector, top_k_cleaned_lyrics)
print(f"\nPrecision for the query '{input_query}': {precision:.2f}")
