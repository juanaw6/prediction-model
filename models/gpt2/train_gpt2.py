from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained("./custom-tokenizer")

vocab_dict = tokenizer.get_vocab()

sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

print("Vocabulary:")
for token, token_id in sorted_vocab:
    print(f"{token_id}: {token}")

config = GPT2Config(
    vocab_size=len(tokenizer),  
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=6,
    n_head=8,
    resid_pdrop=0.2,
    attn_pdrop=0.2,  
    embd_pdrop=0.2,  
    initializer_range=0.02,  
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

training_data = "./data/train/sol_lined_tokens.txt"
dataset = load_dataset("text", data_files={"train": training_data})

split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=False)

print("Training set size:", len(split_dataset["train"]))
print("Test set size:", len(split_dataset["test"]))

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="./results/gpt2",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=200,
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=True,
    warmup_steps=500,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    gradient_accumulation_steps=2,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"The model has {num_params:,} trainable parameters")

trainer.train()

model_save_path = "./results/gpt2/final_model"
model.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")


# # MODEL TEST
# input_text = "<D_1><U_2><D_3>"
# input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
# print(input_ids)

# output = model.generate(
#     input_ids,
#     max_length=len(input_ids[0]) + 1,
#     num_beams=5,
#     early_stopping=True,
# )

# predicted_token = tokenizer.decode(output[0][-1], skip_special_tokens=True)
# print(f"Predicted next token: {predicted_token}")