# Import libraries
import joblib
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

DATA_PATH = 'labeled_songs.joblib'  # Replace with the actual file name
data = joblib.load(DATA_PATH)


# Extract cleaned lyrics and emotion labels
lyrics = data['cleaned_lyrics']
emotions = data['emotion']

# Encode emotions
emotion_classes = sorted(emotions.unique())
emotion_to_int = {emotion: i for i, emotion in enumerate(emotion_classes)}
data['emotion_encoded'] = data['emotion'].map(emotion_to_int)

# Tokenize lyrics
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(lyrics)
sequences = tokenizer.texts_to_sequences(lyrics)

# Pad sequences
max_sequence_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Prepare labels
labels = to_categorical(data['emotion_encoded'], num_classes=len(emotion_classes))

X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# Build the LSTM model
embedding_dim = 64
model = Sequential([
    Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(64),
    Dropout(0.2),
    Dense(len(emotion_classes), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=8,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Evaluate the model
results = model.evaluate(X_val, y_val)
print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")

# Plot training and validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

def recommend_song(emotion, model, tokenizer, data, emotion_to_int):
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


# Test recommendation
print(recommend_song('anger', model, tokenizer, data, emotion_to_int))

model.save('lstm_emotion_model.h5')