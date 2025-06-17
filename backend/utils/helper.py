# helper.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import spacy
import string
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from nltk.corpus import stopwords
from tqdm import tqdm
import spacy.cli

# ============ Setup ============

# Load .env file and HF token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Download and load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ============ Cleaning Function ============

def clean_text(text: str) -> str:
    """Clean text by lemmatizing, removing stopwords, punctuation, and non-alpha tokens."""
    if pd.isna(text):
        return ""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)

# ============ Main Script ============

def main():
    print("📥 Downloading dataset from Hugging Face...")
    dataset_path = hf_hub_download(
        repo_id="FreedomIntelligence/medical-o1-reasoning-SFT",
        filename="medical_o1_sft.json",
        repo_type="dataset",
        token=hf_token
    )

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print("✅ Dataset loaded with shape:", df.shape)
    print(df.head())

    # Clean data
    tqdm.pandas(desc="🔄 Cleaning Question")
    df["instruction_clean"] = df["Question"].progress_apply(clean_text)

    tqdm.pandas(desc="🔄 Cleaning Complex_CoT")
    df["input_clean"] = df["Complex_CoT"].progress_apply(clean_text)

    tqdm.pandas(desc="🔄 Cleaning Response")
    df["output_clean"] = df["Response"].progress_apply(clean_text)

    # Add statistics
    df["output_length"] = df["output_clean"].apply(lambda x: len(x.split()))
    df["instruction_length"] = df["instruction_clean"].apply(lambda x: len(x.split()))

    # Visualize output length
    plt.figure(figsize=(12, 6))
    plt.hist(df["output_length"], bins=50, color="skyblue", edgecolor="black")
    plt.title("Distribution of Output Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), "../../dataset")
    os.makedirs(output_dir, exist_ok=True)

    # Save histogram
    plot_path = os.path.join(output_dir, "output_length_distribution.png")
    plt.savefig(plot_path)
    print(f"📊 Histogram saved to: {plot_path}")
    plt.close()

    # Save cleaned CSV
    output_path = os.path.join(output_dir, "processed_medical_data.csv")
    df[["instruction_clean", "input_clean", "output_clean"]].to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    main()
