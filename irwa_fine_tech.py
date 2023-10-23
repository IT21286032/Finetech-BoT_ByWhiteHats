

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Loading the datasets
product_df = pd.read_csv('dataset/realistic_product_data.csv')
conversational_df = pd.read_csv('dataset/realistic_conversational_data.csv')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    stemmer = nltk.PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

conversational_df['ProcessedQuery'] = conversational_df['UserQuery'].apply(preprocess_text)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(conversational_df['ProcessedQuery'], conversational_df['Intent'], test_size=0.2)

# Building the intent classifier
intent_classifier = make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False), SVC(kernel='linear'))
intent_classifier.fit(X_train, y_train)

def extract_product_name(query):
    for product in product_df['ProductName']:
        if product.lower() in query.lower():
            return product
    return None

def generate_response(user_query):
    # Predict the intent
    intent = intent_classifier.predict([preprocess_text(user_query)])[0]

    # Extract product name if any
    product_name = extract_product_name(user_query)

    if intent == "PriceInquiry" and product_name:
        price = product_df[product_df['ProductName'] == product_name]['Price_LKR'].values[0]
        return f"The price of the {product_name} is LKR {price}."

    elif intent == "ProductDetails" and product_name:
        description = product_df[product_df['ProductName'] == product_name]['Description'].values[0]
        return f"The {product_name} has the following features: {description}."

    elif intent == "ProductAvailability" and product_name:
        return f"Yes, we have the {product_name} in stock."

    else:
        return "I'm sorry, I couldn't understand that. Can you please rephrase or ask something else?"

user_query = "How much is the Samsung Galaxy S31?"
print(generate_response(user_query))

import joblib

# Save the model to disk
joblib.dump(intent_classifier, 'intent_classifier.pkl')