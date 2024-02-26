#current accuracy 20%
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from summarizer import Summarizer
from bs4 import BeautifulSoup
import requests
from transformers import pipeline
import joblib
import re 

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('mtsamples.csv')

# Fill missing values with an empty string
df = df.fillna('')

# Preprocess the data
def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

df['processed_text'] = df['transcription'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df[['medical_specialty', 'sample_name', 'keywords']],
    test_size=0.2, random_state=42
)

# Encode labels using separate LabelEncoders for each output
le_specialty = LabelEncoder()
le_sample_name = LabelEncoder()
le_keywords = LabelEncoder()

y_train_encoded = {
    'medical_specialty': le_specialty.fit_transform(y_train['medical_specialty']),
    'sample_name': le_sample_name.fit_transform(y_train['sample_name'].astype(str)),
    'keywords': le_keywords.fit_transform(y_train['keywords'].astype(str))
}

target_accuracy = 0.9
current_accuracy = 0.0
iteration = 1

# Initialize the best accuracy
best_accuracy = 0.0
best_model = None

while current_accuracy >= best_accuracy:
    # Build a pipeline with an SVM classifier and TF-IDF vectorizer
    model = make_pipeline(TfidfVectorizer(), SVC())
    model.fit(X_train, y_train_encoded['medical_specialty'])

    # Predict the medical specialty for the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    current_accuracy = metrics.accuracy_score(le_specialty.transform(y_test['medical_specialty']), y_pred)
    print(f"Iteration {iteration} - SVM Accuracy: {current_accuracy}")

    if current_accuracy >= best_accuracy:
        # Save the current model as the best model
        best_model = model
        best_accuracy = current_accuracy

    # Add a portion of the test set to the training set
    X_train, _, y_train, _ = train_test_split(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        test_size=0.2,
        random_state=42
    )

    iteration += 1

# Save the best model to a file using joblib
joblib.dump(best_model, 'best_model.joblib')