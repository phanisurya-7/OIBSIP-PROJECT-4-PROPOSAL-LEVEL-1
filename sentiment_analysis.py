import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the datasets
twitter_data = pd.read_csv('twitter_data.csv')  # Replace with the actual file path
play_store_data = pd.read_csv('play_store_data.csv')  # Replace with the actual file path

# Display columns to check data
print("Twitter Data Columns:", twitter_data.columns)
print("Play Store Data Columns:", play_store_data.columns)

# Set up stopwords
stop_words = set(stopwords.words('english'))

# Function to clean and tokenize the text
def clean_text(text):
    if isinstance(text, str):  # Ensure the text is a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
        words = word_tokenize(text)  # Tokenize the text into words
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(words)  # Return the cleaned text as a string
    else:
        return ""  # Return an empty string for non-string values

# Clean Twitter data
twitter_data['cleaned_text'] = twitter_data['clean_text'].apply(clean_text)

# Clean Play Store data (Assuming we are using 'App' column for text analysis)
play_store_data['cleaned_text'] = play_store_data['App'].apply(clean_text)

# Handle missing values in Twitter data
# Remove rows with NaN values in 'cleaned_text' or 'category' columns
twitter_data.dropna(subset=['cleaned_text', 'category'], inplace=True)

# Handle missing values in Play Store data (if required)
# For example, remove NaN in 'cleaned_text' column
play_store_data.dropna(subset=['cleaned_text'], inplace=True)

# Sample data for sentiment analysis (Assuming 'category' is the target for Twitter data)
print("Twitter Data After Cleaning:", twitter_data.head())
print("Play Store Data After Cleaning:", play_store_data.head())

# Split Twitter data into training and testing sets
X = twitter_data['cleaned_text']
y = twitter_data['category']  # Replace with the actual target column name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Classification Report for Twitter Data:")
print(classification_report(y_test, y_pred))
