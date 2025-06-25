import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "cointegrated/rubert-tiny-toxicity"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)


def predict(text):
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    print(f"Тон: {label}, вероятность: {score:.2f}")

    return label, score
