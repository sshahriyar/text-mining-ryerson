import torch
from transformers import pipeline

# Using lightweight GPT-2 just for generating sample text
generator = pipeline("text-generation", model="gpt2")

# Sentiment model to convert text into numeric signal
sentiment_model = pipeline("sentiment-analysis")

def generate_post(prompt):
    # Generate a short text based on prompt
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]

def get_sentiment_score(text):
    # Convert sentiment into simple numeric value
    result = sentiment_model(text)[0]
    return 1.0 if result["label"] == "POSITIVE" else 0.0
