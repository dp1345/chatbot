import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from datetime import datetime

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Load intents.json
with open('intents.json') as file:
    intents = json.load(file)

# Preprocess data
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(text)])

def train_bot(intents):
    corpus = []
    tags = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            corpus.append(preprocess(pattern))
            tags.append(intent['tag'])

    vectorizer = CountVectorizer().fit(corpus)
    X = vectorizer.transform(corpus)

    return vectorizer, X, tags

# Train the bot
vectorizer, X, tags = train_bot(intents)

# Chatbot response
def get_response(user_input):
    processed_input = preprocess(user_input)
    input_vec = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vec, X)
    max_sim_idx = np.argmax(similarities)
    if similarities[0, max_sim_idx] > 0.2:  # Similarity threshold
        tag = tags[max_sim_idx]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that. Can you rephrase?"

def get_time_based_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good morning!"
    elif current_hour < 18:
        return "Good afternoon!"
    else:
        return "Good evening!"

# Streamlit app
st.title("Customer Service Chatbot")

greeting = get_time_based_greeting()
st.subheader(greeting)

user_input = st.text_input("You:")
if user_input:
    response = get_response(user_input)
    st.text_area("Bot:", response, height=100)
