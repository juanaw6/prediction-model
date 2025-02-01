from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
import torch
import os

# Step 1: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Load dataset
dataset = load_dataset("text", data_files={"train": "data.txt"})

# Step 3: Tokenize dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_tokens(["D", "U"])  # Add custom tokens

# Fix: Set padding token
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

def tokenize_function(examples):
    # Tokenize the input and use the same input as labels
    tokenized_input = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=10)
    tokenized_input["labels"] = tokenized_input["input_ids"]  # Add labels for loss calculation
    return tokenized_input

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Load model and move to GPU
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.to(device)  # Move model to GPU

# Step 5: Set up training arguments (enable fp16 for GPU)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Increase batch size for GPU
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training for GPU
    report_to="none"  # Disable reporting to external services
)

# Step 6: Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],
)

# Step 7: Train the model
trainer.train()

# Step 8: Generate predictions
input_text = "DDUDUDU"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)  # Move input to GPU
output = model.generate(input_ids, max_length=len(input_text) + 1)
predicted_token = tokenizer.decode(output[0], skip_special_tokens=True)[-1]

print(f"Predicted next token: {predicted_token}")