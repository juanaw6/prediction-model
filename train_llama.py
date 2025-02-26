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
import os
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] >= 9:
        print("H100 GPU detected, enabling optimizations")
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
    hidden_size=1536,
    intermediate_size=4096,
    num_hidden_layers=16,
    num_attention_heads=16,
    rms_norm_eps=1e-06,
    use_cache=True,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    tie_word_embeddings=False
)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))

model = load_checkpoint_and_dispatch(
    model, 
    None,
    device_map="auto",
    no_split_module_classes=["LlamaDecoderLayer"]
)

print(f"Model embedding size: {model.get_output_embeddings().weight.shape}")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

tokenized_dataset = split_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    num_proc=8
)

tokenized_dataset.save_to_disk("./tokenized_dataset")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./results/llama_medium_temp",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=3e-4,
    weight_decay=0.01,
    bf16=True,
    warmup_steps=500,
    report_to="tensorboard",
    lr_scheduler_type="cosine",
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    torch_compile=True,
    optim="adamw_torch_fused",
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

final_model_path = "./results/llama_medium/final_model"
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Final model saved to {final_model_path}")

try:
    from optimum.bettertransformer import BetterTransformer
    optimized_model = BetterTransformer.transform(model)
    optimized_model.save_pretrained(f"{final_model_path}_optimized")
    print(f"Optimized model saved to {final_model_path}_optimized")
except ImportError:
    print("optimum package not found. Skipping optimized model conversion.")

print("Training and model saving completed successfully.")