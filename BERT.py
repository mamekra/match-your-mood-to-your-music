import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

data = pd.read_csv('labeled_songs.csv')
# Preview the dataset
data.head()

data=data.drop(columns=['year','lyrics'])
data['emotion'] = data['emotion'].fillna('neutral')

# Extract cleaned lyrics and emotion labels
lyrics = data['cleaned_lyrics']
emotions = data['emotion']

# Convert emotions to strings before sorting
emotions = emotions.astype(str)

# Encode emotions as integers
emotion_classes = sorted(emotions.unique())
emotion_to_int = {emotion: i for i, emotion in enumerate(emotion_classes)}
data['emotion_encoded'] = data['emotion'].map(emotion_to_int)

# Split into train and validation sets
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    lyrics, data['emotion_encoded'], test_size=0.2, random_state=42
)
# Check dataset stats
print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Handle class imbalance: Oversampling minority classes
train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
balanced_train_df = train_df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(train_df['label'].value_counts().max(), replace=True)
).reset_index(drop=True)

train_texts = balanced_train_df['text'].tolist()
train_labels = balanced_train_df['label'].tolist()

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Increase max_length to capture more context
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Check an example tokenized input
print(train_encodings.keys())

# Create a custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        return item

train_dataset = EmotionDataset(train_encodings, list(train_labels))
val_dataset = EmotionDataset(val_encodings, list(val_labels))

# Check dataset size
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Load the model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(emotion_classes)
)


training_args = TrainingArguments(
    output_dir='./BERTresults',          # Output directory for checkpoints
    num_train_epochs=4,              # Fewer epochs
    learning_rate=2e-5,
    per_device_train_batch_size=64,   # Smaller batch size
    weight_decay=0.01,               # Regularization
    logging_dir='./logs',            # Log directory
    logging_steps=10,
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
# Train the model
trainer.train()

# Evaluate the model
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import torch # Import torch for softmax

# Predictions
preds = trainer.predict(val_dataset)
y_true = preds.label_ids

# Convert logits to probabilities using softmax
probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()

# Get predicted labels
y_pred = np.argmax(preds.predictions, axis=1)

# Classification Report
print(classification_report(y_true, y_pred, target_names=emotion_classes))

# ROC-AUC (using probabilities)
roc_auc = roc_auc_score(y_true, probs, multi_class='ovr')
print(f'ROC-AUC: {roc_auc}')

# Confusion matrix
cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_classes, yticklabels=emotion_classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
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
