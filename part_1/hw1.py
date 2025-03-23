from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime

class User:
    def __init__(self, login: str, password: str, balance:int=0):
        self.login = login
        self.password = password
        self.balance = balance
        self.history = []

    def has_enough_tokens(self, required_tokens:int) -> bool:
        return self.balance >= required_tokens

    def deduct_tokens(self, tokens:int):
        if self.has_enough_tokens(tokens):
            self.balance -= tokens
            return True
        return False

    def add_to_history(self, text:str, result:str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({
            "timestamp": timestamp,
            "text": text,
            "sentiment": result
        })


class SentimentModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
        labels = ["negative", "neutral", "positive"]
        return labels[scores.argmax()]


class SentimentAnalysisTask:
    def __init__(self, user: str, text: str, model: str):
        self.user = user
        self.text = text
        self.model = model

    def execute(self):
        if self.user.deduct_tokens(10):
            result = self.model.analyze_sentiment(self.text)
            self.user.add_to_history(self.text, result)
            return result
        else:
            return "Not enough tokens"


# Пример использования
if __name__ == "__main__":
    user = User("example_user", "password123", balance=20)
    model = SentimentModel("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    text = "US removes sanctions against Russian crypto service facilitating sanctions circumvention"
    task = SentimentAnalysisTask(user, text, model)
    result = task.execute()
    print(f"Sentiment: {result}")
    print(f"User balance: {user.balance}")
    print(f"User history: {user.history}")
