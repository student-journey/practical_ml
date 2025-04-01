from models.event import *


if __name__ == "__main__":
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    text = "US removes sanctions against Russian crypto service facilitating sanctions circumvention"

    model = SentimentModel(model_name)
    user = User("example_user", "password123", 100)
    
    task = SentimentAnalysisTask(user, model)
    result = task.execute(text)

    print(result)
