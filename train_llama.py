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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.95)

training_data = "./sol_lined_tokens.txt"
dataset = load_dataset("text", data_files={"train": training_data})
split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=False)
print("Training set size:", len(split_dataset["train"]))
print("Test set size:", len(split_dataset["test"]))

tokenizer = AutoTokenizer.from_pretrained("./custom-tokenizer")

config = LlamaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=128,
    hidden_size=2048,       # Increased from 1536
    intermediate_size=5632, # Increased from 4096
    num_hidden_layers=24,   # Increased from 16
    num_attention_heads=32, # Increased from 16
    rms_norm_eps=1e-06,
    use_cache=False,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    tie_word_embeddings=False
)

model = AutoModelForCausalLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print(f"Model embedding size: {model.get_output_embeddings().weight.shape}")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=40)

tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./results/llama_medium_temp",
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    num_train_epochs=12,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=1e-4,
    weight_decay=0.05,
    bf16=True,
    warmup_steps=500,
    report_to="tensorboard",
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=False,
    torch_compile=True,
    optim="adamw_torch_fused",
    max_grad_norm=1.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

torch.cuda.empty_cache()
gc.collect()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"The model has {num_params:,} trainable parameters")

trainer.train()

model_save_path = "./results/llama_medium/final_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")