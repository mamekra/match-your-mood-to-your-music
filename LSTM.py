# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#import data
data= pd.read_csv('labeled_songs.csv')

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

#train test split
X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# Build the LSTM model
embedding_dim = 32
model = Sequential([
    Embedding(input_dim=5000, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(16, kernel_regularizer=l2(0.01)),  # Reduced LSTM units, added L2 regularization
    Dropout(0.5),  # Increased dropout
    Dense(len(emotion_classes), activation='softmax', kernel_regularizer=l2(0.01))  # L2 regularization
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

#model evaluation and plot

# Get predictions
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

print(classification_report(y_true, y_pred, target_names=emotion_classes))


accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')


# For multiclass ROC-AUC, we need the probability predictions
roc_auc = roc_auc_score(y_val, y_pred_probs, multi_class='ovr')
print(f'ROC-AUC: {roc_auc}')


# Get predictions
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

# Create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_classes, yticklabels=emotion_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model training/validation loss and accuracy
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

# Recommendation function
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