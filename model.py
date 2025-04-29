import pandas as pd
import numpy as np
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Display first few rows
print("First 5 entries:")
print(data.head())

# Checking columns
print("\nColumns in dataset:")
print(data.columns)

# Keeping only necessary columns (assuming first two columns are label and message)
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# 2. Preprocessing

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing
data['message'] = data['message'].apply(preprocess_text)

# Encode labels
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])  # ham -> 0, spam -> 1

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 4. Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Model Training
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. Prediction and Evaluation
y_pred = model.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
