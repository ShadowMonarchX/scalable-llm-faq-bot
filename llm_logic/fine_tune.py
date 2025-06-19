import sys
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch

# ---------- Config ---------- #
model_name = "EleutherAI/gpt-neo-125M"
default_data_path = "dataset/medical_o1_reasoning_sft/fine_tune_data.jsonl"
output_dir = "./models/gpt_neo_finetuned"
logging_dir = "./logs"

# ---------- CLI Argument ---------- #
data_path = sys.argv[1] if len(sys.argv) > 1 else default_data_path
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at path: {data_path}")

# ---------- Load Model & Tokenizer ---------- #
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ---------- Load Dataset ---------- #
print(f"Loading dataset from: {data_path}")
dataset = load_dataset("json", data_files=data_path, split="train")

# ---------- Tokenization ---------- #
def tokenize_function(example):
    return tokenizer(
        f"### Question:\n{example['prompt']}\n\n### Answer:\n{example['response']}",
        padding="max_length",
        truncation=True,
        max_length=512
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])

# ---------- Split into Train/Validation ---------- #
train_test = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# ---------- Data Collator ---------- #
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------- TrainingArguments ---------- #
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir=logging_dir,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",  # Set to "wandb" if using wandb
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True
)

# ---------- Trainer ---------- #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ---------- Train ---------- #
print("Starting training...")
trainer.train()

# ---------- Save Model ---------- #
print(f"Saving fine-tuned model to: {output_dir}/final")
trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")
print("Training complete and model saved.")
