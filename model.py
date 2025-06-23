import torch
import time
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

text = "Ты сволоч"

start_time = time.time()

result = classifier(text)[0]

end_time = time.time()

label = result["label"]
score = result["score"]
elapsed_time = end_time - start_time

print(f"Тон: {label}, вероятность: {score:.2f}")
print(f"Время ответа: {elapsed_time:.4f} секунд")