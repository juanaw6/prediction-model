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

torch.set_float32_matmul_precision('high')

training_data = "./data/train/sol_lined_tokens.txt"
dataset = load_dataset("text", data_files={"train": training_data})
split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=False)
print("Training set size:", len(split_dataset["train"]))
print("Test set size:", len(split_dataset["test"]))

tokenizer = AutoTokenizer.from_pretrained("./custom-tokenizer")


config = LlamaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=64,
    hidden_size=768,
    intermediate_size=2176,
    num_hidden_layers=8,
    num_attention_heads=12,
    rms_norm_eps=1e-06,
    use_cache=True,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id,
    tie_word_embeddings=True
)


model = AutoModelForCausalLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print(f"Model embedding size: {model.get_output_embeddings().weight.shape}")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./results/llama_small",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    num_train_epochs=20,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=5,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=3e-4,
    weight_decay=0.01,
    fp16=True,
    warmup_steps=250,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    lr_scheduler_type="cosine",
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

model_save_path = "./results/llama_small/final_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")