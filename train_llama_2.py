import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    LlamaConfig
)

training_data = "./data/train/sol_lined.txt"
dataset = load_dataset("text", data_files={"train": training_data})

split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=False)

print("Training set size:", len(split_dataset["train"]))
print("Test set size:", len(split_dataset["test"]))

tokenizer = AutoTokenizer.from_pretrained("./custom-tokenizer")

from transformers import LlamaConfig

config = LlamaConfig(
    vocab_size=len(tokenizer),  # Correct vocab size
    hidden_size=2048,  # Increased for better representation
    intermediate_size=4 * 2048,  # Standard MLP size
    num_hidden_layers=12,  # Reduced for small dataset
    num_attention_heads=32,  # Increased for better attention
    rms_norm_eps=1e-05,  # Improved numerical stability
    use_cache=False,  # Disable cache during training
    rope_scaling={"type": "linear", "factor": 2.0},  # Improves long-sequence learning
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    tie_word_embeddings=True,  # Reduces parameters, better efficiency
)

model = AutoModelForCausalLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print(f"Model embedding size: {model.get_output_embeddings().weight.shape}")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./results/llama_2",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=6e-4,
    weight_decay=0.01,
    fp16=True,
    warmup_steps=250,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    lr_scheduler_type="cosine",
    torch_compile=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

torch.cuda.empty_cache()
gc.collect()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"The model has {num_params:,} trainable parameters")

trainer.train()

model_save_path = "./results/llama_2/final_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")