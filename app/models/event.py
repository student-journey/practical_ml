from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import hashlib
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def analyze(self, text: str) -> str:
        pass

class SentimentModel(BaseModel):
    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze(self, text: str) -> str:
        inputs = self._tokenizer(text, return_tensors="pt")
        scores = self._model(**inputs).logits.softmax(dim=1).detach().numpy()[0]
        return ["negative", "neutral", "positive"][scores.argmax()]

class Account:
    def __init__(self, initial_balance: int = 0):
        self._balance = initial_balance

    def deduct_tokens(self, amount: int) -> bool:
        if self._balance >= amount:
            self._balance -= amount
            return True
        return False

    @property
    def balance(self) -> int:
        return self._balance

class TransactionHistory:
    def __init__(self):
        self._transactions = []

    def add_transaction(self, text: str, result: str):
        self._transactions.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "sentiment": result
        })

    @property
    def history(self) -> list:
        return self._transactions

class User:
    def __init__(self, login: str, password: str, initial_balance: int = 0):
        self.login = login
        self._hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.account = Account(initial_balance)
        self.history = TransactionHistory()

class AnalysisTask(ABC):
    COST = 10

    def __init__(self, user: User, model: BaseModel):
        self.user = user
        self.model = model

    @abstractmethod
    def execute(self, text: str) -> str:
        pass

class SentimentAnalysisTask(AnalysisTask):
    def execute(self, text: str) -> str:
        if not self.user.account.deduct_tokens(self.COST):
            return "Недостаточно токенов"
        
        result = self.model.analyze(text)
        self.user.history.add_transaction(text, result)
        return result