# Match Your Mood with Your Music

Song lyrics often carry deep sentimental meaning, making music a powerful way to express emotions. This project aims to develop a song recommendation system that suggests songs based on a chosen emotion. To accomplish this, sentiment analysis was performed to categorize the lyrics of 100,000 songs from the [Genius Song Lyrics Dataset on Kaggle](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) into different emotional groups.

### Data Processing

- **Data Cleaning:** The initial dataset is preprocessed with `clean_dataset.py`, which filters non-English lyrics, removes noise, and normalizes text.
- **Emotion Labeling:** The NRCLex Python package is used to analyze text and assign predefined emotions. This process is handled in `emotion_labeling_nrc.py`, creating a labeled dataset for further model training.

### Models Used

#### Traditional Machine Learning Models:
- **Logistic Regression** in `tf-idf_logistic_regression.py`
- **Naïve Bayes** in `tf-idf_naive_bayes.py`
- **Support Vector Machine (SVM)** in `tf-idf_svm.py`

#### Deep Learning Models:
- **Long Short-Term Memory (LSTM)** in `LSTM.py`
- **Gated Recurrent Units (GRU)** in `gru_with_overfitting.py` and `gru_without_overfitting.py`
- **DistilBERT (a lightweight version of BERT)** in `BERT.py`

Each model was trained and tested to evaluate its ability to classify emotions from song lyrics. The objective was to determine which model provides the most accurate and reliable emotion predictions. The results are visualized using histogram generated with `histogram_creation.py`, showcasing model performance metrics such as accuracy and auc-roc.

### Summary

This project explores data preprocessing steps for sentiment analysis, emotion labeling techniques, model architectures, and performance evaluations. The trained models are leveraged to recommend songs based on the user's selected emotion, forming a comprehensive sentiment-based song recommendation system.
