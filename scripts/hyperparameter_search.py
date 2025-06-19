from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Load tokenizer and model
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("json", data_files="dataset/fine_tune_data.jsonl", split="train")

# Tokenize
def tokenize(example):
    return tokenizer(
        f"### Question:\n{example['prompt']}\n\n### Answer:\n{example['response']}",
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["prompt", "response"])
train_val = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val["train"]
tokenized_val_dataset = train_val["test"]

# Tuning configs
learning_rates = [3e-5, 5e-5, 1e-4]
epoch_counts = [3, 4, 5]

for lr in learning_rates:
    for epochs in epoch_counts:
        print(f"\n🧪 Training with LR={lr}, Epochs={epochs}")
        training_args = TrainingArguments(
            output_dir=f"./models/gpt-finetuned-lr{lr}-ep{epochs}",
            per_device_train_batch_size=4,
            num_train_epochs=epochs,
            learning_rate=lr,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_dir="./logs",
            logging_steps=50,
            save_total_limit=2
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=tokenized_val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )

        trainer.train()
