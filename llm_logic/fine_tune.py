import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase
)
from typing import List, Dict, Union
from torch.nn.utils.rnn import pad_sequence

# ------------------ Config ------------------ #
model_name = "EleutherAI/gpt-neo-125M"
data_path = "/content/fine_tune_data.jsonl"
output_dir = "./neo_outputs"
logging_dir = "./neo_logs"

# ------------------ Load Model & Tokenizer ------------------ #
try:
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
except Exception as e:
    raise RuntimeError(f"Failed to load model/tokenizer: {e}")

# ------------------ Load Dataset ------------------ #
try:
    abs_path = os.path.abspath(data_path)
    print(f"Loading dataset from: {abs_path}")
    df = pd.read_json(abs_path, lines=True)
    dataset = Dataset.from_pandas(df)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}")

# ------------------ Tokenization ------------------ #
def tokenize_function(example):
    try:
        text = f"### Question:\n{example['prompt']}\n\n### Answer:\n{example['response']}"
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"]
        }
    except Exception as e:
        print(f"Tokenization failed for example: {example} — {e}")
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

try:
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, remove_columns=["prompt", "response"])
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
except Exception as e:
    raise RuntimeError(f"Tokenization or filtering failed: {e}")

# ------------------ Train / Eval Split ------------------ #
try:
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
except Exception as e:
    raise RuntimeError(f"Dataset splitting failed: {e}")

# ------------------ Custom Data Collator ------------------ #
class CausalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if len(f["input_ids"]) > 0]
        if len(features) == 0:
            raise ValueError("No valid sequences in batch.")

        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

data_collator = CausalDataCollator(tokenizer)

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

# ------------------ Sample Batch Check ------------------ #
print("Checking a sample batch from the data collator...")
try:
    for i, batch in enumerate(trainer.get_train_dataloader()):
        if i == 0:
            for k, v in batch.items():
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            break
except Exception as e:
    print(f"Error during batch inspection: {e}")

# ------------------ Training ------------------ #
print("Starting training...")
try:
    trainer.train()
except Exception as e:
    raise RuntimeError(f"Training failed: {e}")

# ------------------ Save Model ------------------ #
try:
    final_output_path = os.path.join(output_dir, "final")
    print(f"Saving model to: {final_output_path}")
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    print("Training complete.")
except Exception as e:
    raise RuntimeError(f"Saving model failed: {e}")
