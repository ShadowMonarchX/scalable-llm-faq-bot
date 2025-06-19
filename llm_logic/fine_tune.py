import sys
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ------------------ Config ------------------ #
model_name = "EleutherAI/gpt-neo-125M"
default_data_path = "dataset/medical_o1_reasoning_sft/fine_tune_data.jsonl"
output_dir = "./models/gpt_neo_finetuned"
logging_dir = "./logs"

# ------------------ CLI Dataset Path ------------------ #
data_path = sys.argv[1] if len(sys.argv) > 1 else default_data_path
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

# ------------------ Load Model & Tokenizer ------------------ #
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# ------------------ Optional: Check GPU BF16 support ------------------ #
use_4bit = False
compute_dtype = torch.float16
if compute_dtype == torch.float16 and use_4bit and torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# ------------------ Load & Tokenize Dataset ------------------ #
print(f"Loading dataset from: {data_path}")
dataset = load_dataset("json", data_files=data_path, split="train")

def tokenize_function(example):
    tokenized = tokenizer(
        f"### Question:\n{example['prompt']}\n\n### Answer:\n{example['response']}",
        truncation=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",  # Ensures tensor output
    )

    # Convert from tensor -> list for HF Trainer compatibility
    return {
        "input_ids": tokenized["input_ids"][0].tolist(),
        "attention_mask": tokenized["attention_mask"][0].tolist(),
        "labels": tokenized["input_ids"][0].tolist()
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])

# ------------------ Train / Eval Split ------------------ #
train_test = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# ------------------ Data Collator ------------------ #
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ------------------ Training Arguments ------------------ #
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir=logging_dir,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    do_train=True,
    do_eval=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# ------------------ Trainer Setup ------------------ #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ------------------ Train ------------------ #
print("Starting training...")
trainer.train()

# ------------------ Save ------------------ #
final_output_path = os.path.join(output_dir, "final")
print(f"Saving model to: {final_output_path}")
trainer.save_model(final_output_path)
tokenizer.save_pretrained(final_output_path)
print("Training complete.")