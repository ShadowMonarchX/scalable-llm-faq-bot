
import json
import random
import os

def split_jsonl_dataset(input_path, train_path, val_path, val_ratio=0.1, seed=42):
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    random.seed(seed)
    random.shuffle(records)

    split_idx = int(len(records) * (1 - val_ratio))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    with open(train_path, 'w', encoding='utf-8') as f_train:
        for r in train_records:
            f_train.write(json.dumps(r) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f_val:
        for r in val_records:
            f_val.write(json.dumps(r) + '\n')

    print(f"Train records: {len(train_records)}, Validation records: {len(val_records)}")

# Usage
split_jsonl_dataset(
    input_path="dataset/fine_tune_data.jsonl",
    train_path="dataset/train.jsonl",
    val_path="dataset/val.jsonl"
)

