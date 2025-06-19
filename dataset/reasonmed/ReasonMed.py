

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
import nltk
import spacy
import spacy.cli
from nltk.corpus import stopwords

# from dotenv import load_dotenv


# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")



nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """Basic NLP cleaning: lemmatization, lowercase, stopwords removal"""
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
    """Filters samples with very short questions or outputs"""
    return len(row["instruction_clean"].split()) >= 3 and len(row["output_clean"].split()) >= 3


def main():
    print("Loading ReasonMed dataset...")
    dataset = load_dataset("lingshu-medical-mllm/ReasonMed")
    df = dataset["train"].to_pandas()
    print(f"Original shape: {df.shape}")

    # Renaming for consistency
    df = df.rename(columns={"instruction": "Question", "input": "Complex_CoT", "output": "Response"})
    df.dropna(subset=["Question", "Response"], inplace=True)
    df.drop_duplicates(subset=["Question", "Response"], inplace=True)

    tqdm.pandas(desc="Cleaning Question")
    df["instruction_clean"] = df["Question"].progress_apply(clean_text)

    tqdm.pandas(desc="Cleaning Complex_CoT")
    df["input_clean"] = df["Complex_CoT"].progress_apply(clean_text)

    tqdm.pandas(desc="Cleaning Response")
    df["output_clean"] = df["Response"].progress_apply(clean_text)

    # Token Lengths
    df["instruction_length"] = df["instruction_clean"].apply(lambda x: len(x.split()))
    df["input_length"] = df["input_clean"].apply(lambda x: len(x.split()))
    df["output_length"] = df["output_clean"].apply(lambda x: len(x.split()))

    df = df[df.apply(filter_valid_rows, axis=1)]
    print(f"After filtering: {df.shape}")
    print(df[["instruction_length", "input_length", "output_length"]].describe())

    output_dir = os.path.join(os.getcwd(), "dataset", "reasonmed")

    os.makedirs(output_dir, exist_ok=True)

    def save_plot(fig, filename):
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        print(f" Saved: {filename}")

    # Histograms
    for col, title, color in [
        ("instruction_length", "Instruction", "lightgreen"),
        ("input_length", "Input (Complex_CoT)", "salmon"),
        ("output_length", "Output (Response)", "skyblue")
    ]:
        fig = plt.figure(figsize=(12, 6))
        sns.histplot(df[col], bins=50, kde=True, color=color)
        plt.title(f"Distribution of {title} Lengths")
        save_plot(fig, f"{col}_distribution.png")

    # Instruction vs Output Length
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x="instruction_length", y="output_length", data=df, alpha=0.4)
    plt.title("Instruction vs Output Length")
    save_plot(fig, "instruction_vs_output.png")

    # Correlation Heatmap
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(df[["instruction_length", "input_length", "output_length"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Length Correlation Heatmap")
    save_plot(fig, "length_correlation.png")

    # Most Common Output Tokens
    top_tokens = Counter(" ".join(df["output_clean"]).split()).most_common(30)
    tokens, freqs = zip(*top_tokens)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=list(freqs), y=list(tokens), palette="mako")
    plt.title("Top 30 Frequent Lemmas in Output")
    save_plot(fig, "frequent_output_tokens.png")

    # Ratio Plot
    df["instruction_output_ratio"] = df["instruction_length"] / (df["output_length"] + 1)
    fig = plt.figure(figsize=(10, 5))
    sns.histplot(df["instruction_output_ratio"], bins=50, kde=True, color="purple")
    plt.title("Instruction-to-Output Length Ratio")
    save_plot(fig, "instruction_output_ratio.png")


    df.to_csv(os.path.join(output_dir, "cleaned_reasonmed.csv"), index=False)
    print("Saved CSV: cleaned_reasonmed.csv")

    with open(os.path.join(output_dir, "fine_tune_reasonmed.jsonl"), "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json.dump({"prompt": row["instruction_clean"], "response": row["output_clean"]}, f)
            f.write("\n")
    print("Saved JSONL: fine_tune_reasonmed.jsonl")

if __name__ == "__main__":
    main()
