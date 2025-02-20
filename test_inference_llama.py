import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch
from sklearn.metrics import classification_report

model_path = "./results/llama/final_model"
tokenizer = AutoTokenizer.from_pretrained("./custom-tokenizer")

loaded_model = AutoModelForCausalLM.from_pretrained(model_path)

loaded_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(loaded_device)

df = pd.read_csv("./data/test/testset.csv")
input_sequences = df["input_sequence"].tolist()
actual_targets = df["target"].tolist()

def predict_next_tokens(input_sequences):
    predictions = []
    for input_text in input_sequences:
        inputs = tokenizer(input_text, return_tensors="pt").to(loaded_device)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            predicted_token_id = torch.argmax(next_token_logits, dim=-1)
            predicted = tokenizer.decode(predicted_token_id, skip_special_tokens=True)

        predictions.append(predicted)
        print(f"Input: {input_text} | Predicted: {predicted}")

    return predictions

def clean_targets(targets, predictions):
    cleaned_targets = []
    cleaned_predictions = []
    for i in range(len(targets)):
        if isinstance(targets[i], str) and isinstance(predictions[i], str):
            cleaned_targets.append(targets[i])
            cleaned_predictions.append(predictions[i])
    return cleaned_targets, cleaned_predictions


predictions = predict_next_tokens(input_sequences)
actual_targets, predictions = clean_targets(actual_targets, predictions)  # remove non-strings
report = classification_report(actual_targets, predictions, digits=4, zero_division=1) # zero_division added
print(report)