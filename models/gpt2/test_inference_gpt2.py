import pandas as pd
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
    GPT2LMHeadModel
)
import torch
from sklearn.metrics import classification_report

loaded_model = GPT2LMHeadModel.from_pretrained("./results/gpt2/final_model")
tokenizer = PreTrainedTokenizerFast.from_pretrained("./custom-tokenizer")

vocab_dict = tokenizer.get_vocab()
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
print("Vocabulary:")
for token, token_id in sorted_vocab:
    print(f"{token_id}: {token}")

loaded_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(loaded_device)

df = pd.read_csv("./data/test/test_dataset.csv")
input_sequences = df["input_sequence"].tolist()
actual_targets = df["target"].tolist()

def predict_next_tokens(input_sequences):
    predictions = []
    for input_text in tqdm(input_sequences, desc="Processing", unit="sample"):

        inputs = tokenizer.encode(input_text, return_tensors="pt").to(loaded_device)
        
        outputs = loaded_model.generate(
            inputs,
            max_length=len(inputs[0]) + 1,
            num_beams=1,
            early_stopping=True,
        )
        
        predicted = tokenizer.decode(outputs[0][-1], skip_special_tokens=True)
        predictions.append(predicted)
        # print(f"Input: {input_text} | Predicted: {predicted}")
    
    return predictions

predictions = predict_next_tokens(input_sequences)
report = classification_report(actual_targets, predictions, digits=4)
print(report)