import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load datasets
twitter_data = pd.read_csv('data/twitter_sentiment.csv')
play_store_data = pd.read_csv('data/play_store_reviews.csv')

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    return ''

twitter_data['cleaned_text'] = twitter_data['text'].apply(preprocess_text)
play_store_data['cleaned_text'] = play_store_data['review'].apply(preprocess_text)
vectorizer = CountVectorizer(max_features=5000)
X_twitter = vectorizer.fit_transform(twitter_data['cleaned_text']).toarray()
y_twitter = twitter_data['sentiment']

X_play_store = vectorizer.fit_transform(play_store_data['cleaned_text']).toarray()
y_play_store = play_store_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X_twitter, y_twitter, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# Visualize class distribution
sns.countplot(x='sentiment', data=twitter_data)
plt.title('Twitter Sentiment Distribution')
plt.show()
