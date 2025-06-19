import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from nltk.corpus import stopwords
from tqdm import tqdm
import spacy.cli

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Clean text by lemmatizing, removing stopwords, punctuation, and non-alpha tokens.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)


def filter_valid_rows(row) -> bool:
    """
    Filter out rows with short or empty question or answer texts.
    """
    if len(row["instruction_clean"].split()) < 3:
        return False
    if len(row["output_clean"].split()) < 3:
        return False
    return True


def main():
    print("Downloading dataset from Hugging Face...")
    dataset_path = hf_hub_download(
        repo_id="FreedomIntelligence/medical-o1-reasoning-SFT",
        filename="medical_o1_sft.json",
        repo_type="dataset",
        token=hf_token
    )

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Dataset loaded with shape: {df.shape}")
    print(df.head())

    # Clean text columns
    tqdm.pandas(desc="Cleaning Question")
    df["instruction_clean"] = df["Question"].progress_apply(clean_text)

    tqdm.pandas(desc="Cleaning Complex_CoT")
    df["input_clean"] = df["Complex_CoT"].progress_apply(clean_text)

    tqdm.pandas(desc="Cleaning Response")
    df["output_clean"] = df["Response"].progress_apply(clean_text)

    # Calculate text lengths for analysis
    df["instruction_length"] = df["instruction_clean"].apply(lambda x: len(x.split()))
    df["input_length"] = df["input_clean"].apply(lambda x: len(x.split()))
    df["output_length"] = df["output_clean"].apply(lambda x: len(x.split()))

    # Filter rows with short or empty texts
    df = df[df.apply(filter_valid_rows, axis=1)]
    print(f"Filtered dataset shape after removing short entries: {df.shape}")

    print("Summary statistics for text lengths:")
    print(df[["instruction_length", "input_length", "output_length"]].describe())


    output_dir = os.path.join(os.getcwd(), "dataset", "medical_o1_reasoning_sft")

    os.makedirs(output_dir, exist_ok=True)

    # Visualization 1: Output length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df["output_length"], bins=50, color="skyblue", kde=True)
    plt.title("Distribution of Output Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "output_length_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

    # Visualization 2: Instruction (Question) length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df["instruction_length"], bins=50, color="lightgreen", kde=True)
    plt.title("Distribution of Instruction (Question) Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "instruction_length_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

    # Visualization 3: Input (Complex_CoT) length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df["input_length"], bins=50, color="salmon", kde=True)
    plt.title("Distribution of Input (Complex_CoT) Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "input_length_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

    # Visualization 4: Scatter plot Instruction length vs Output length
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="instruction_length", y="output_length", data=df, alpha=0.4)
    plt.title("Instruction Length vs Output Length")
    plt.xlabel("Instruction Length (words)")
    plt.ylabel("Output Length (words)")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "instruction_vs_output_length.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

    # Visualization 5: Correlation heatmap of lengths
    plt.figure(figsize=(6, 5))
    corr = df[["instruction_length", "input_length", "output_length"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Between Text Lengths")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "lengths_correlation_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

    # Prepare dataset for fine-tuning: JSONL with prompt-response pairs
    fine_tune_records = []
    for _, row in df.iterrows():
        prompt_text = row["instruction_clean"]
        response_text = row["output_clean"]
        fine_tune_records.append({
            "prompt": prompt_text,
            "response": response_text
        })

    jsonl_path = os.path.join(output_dir, "fine_tune_data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for record in fine_tune_records:
            f_out.write(json.dumps(record) + "\n")
    print(f"Fine-tuning dataset saved in JSONL format at: {jsonl_path}")

    # Save cleaned DataFrame as CSV for reference
    csv_path = os.path.join(output_dir, "cleaned_medical_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Cleaned CSV saved at: {csv_path}")


if __name__ == "__main__":
    main()
