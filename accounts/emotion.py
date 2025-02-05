
import os
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch


emotion_model_name = "ahmettasdemir/distilbert-base-uncased-finetuned-emotion"
emotion_model = DistilBertForSequenceClassification.from_pretrained(emotion_model_name)
emotion_tokenizer = DistilBertTokenizer.from_pretrained(emotion_model_name)


sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_name)


def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors='pt')
    outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_emotion = torch.argmax(probs, dim=1).item()
    return emotion_model.config.id2label[predicted_emotion]


def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors='pt')
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_sentiment = torch.argmax(probs, dim=1).item()
    return sentiment_model.config.id2label[predicted_sentiment]