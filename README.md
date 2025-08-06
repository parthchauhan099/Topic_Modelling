# Product Review - Topic Modeling and Classification

This project demonstrates a hybrid approach of **unsupervised topic modeling** and **supervised deep learning** to classify Amazon product reviews. The goal is to extract meaningful topics from review texts and then build an LSTM model to automatically categorize new reviews based on these topics.

---

## 🚀 Project Highlights

- **Dataset:** product dataset (title and text fields).
- **NLP Techniques:** 
  - Text preprocessing using regex, contractions, lemmatization, and POS filtering.
  - TF-IDF feature extraction.
- **Topic Modeling:** 
  - Applied **Non-Negative Matrix Factorization (NMF)** to extract latent topics.
  - Identified product categories like:
    - Kindle / Tablets
    - Headphones / Buds
    - Speakers / Alexa
    - TV / Roku
- **Deep Learning:**
  - Used **LSTM** neural network for multi-class classification of reviews.
  - Implemented using TensorFlow and Keras.

---

## 📊 Workflow

1. **Data Preprocessing**
   - Cleaning, lemmatization, and POS tagging
   - Sentiment analysis (TextBlob)

2. **Exploratory Data Analysis**
   - Word clouds and frequency distribution
   - N-gram analysis (unigrams, bigrams, trigrams)

3. **Topic Modeling with NMF**
   - TF-IDF vectorization
   - Topic-word distributions
   - Mapping topics to product categories

4. **LSTM Classification**
   - Tokenization and padding
   - Categorical encoding of topics
   - Sequential LSTM model with embedding

---

## 🧰 Tech Stack

- **Languages:** Python
- **Libraries:** Pandas, NumPy, scikit-learn, SpaCy, TextBlob, TensorFlow, Keras, Matplotlib, Seaborn, WordCloud
- **Techniques:** NMF, LSTM, TF-IDF, Tokenization, Sentiment Analysis

---
