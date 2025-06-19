# # pip install peft accelerate bitsandbytes
# pip install transformers datasets accelerate bitsandbytes peft
# pip install transformers datasets accelerate bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load a lightweight open model (no login/token needed)
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenization
def tokenize_function(example):
    return tokenizer(
        f"### Question:\n{example['prompt']}\n\n### Answer:\n{example['response']}",
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Load from the JSONL file
dataset = load_dataset("json", data_files="dataset/fine_tune_data.jsonl", split="train")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./gpt-finetuned-customer-support",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",  # Or "epoch" if you split validation
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Start training
trainer.train()



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


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned model
model_path = "./gpt-finetuned-customer-support"  # Path to your trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

def generate_response(prompt, max_tokens=100):
    input_text = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Answer:")[-1].strip()

# Load validation examples
val_samples = []
with open("dataset/val.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        val_samples.append(json.loads(line))

# Generate and compare
results = []
for sample in val_samples[:10]:  # Limit to 10 for demo
    prompt = sample['prompt']
    expected = sample['response']
    predicted = generate_response(prompt)
    results.append((prompt, expected, predicted))
    print(f"\n🟡 Question: {prompt}")
    print(f"✅ Expected: {expected}")
    print(f"🔷 Predicted: {predicted}")


from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = []

for _, expected, predicted in results:
    score = scorer.score(expected, predicted)
    scores.append(score['rougeL'].fmeasure)

avg_rouge = sum(scores) / len(scores)
print(f"\n📊 Average ROUGE-L F1 Score: {avg_rouge:.4f}")


custom_query = "How do I change my phone's ringtone?"
response = generate_response(custom_query)
print(f"\n📨 Custom Query: {custom_query}")
print(f"🤖 Model Response: {response}")


from transformers import TrainingArguments

# Define multiple configurations
learning_rates = [3e-5, 5e-5, 1e-4]
epoch_counts = [3, 4, 5]

for lr in learning_rates:
    for epochs in epoch_counts:
        print(f"\n🧪 Training with LR={lr}, Epochs={epochs}")
        training_args = TrainingArguments(
            output_dir=f"./gpt-finetuned-lr{lr}-ep{epochs}",
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
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )

        trainer.train()



from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model
tokenizer = AutoTokenizer.from_pretrained("gpt-finetuned-customer-support")
model = AutoModelForCausalLM.from_pretrained("gpt-finetuned-customer-support").eval()

class Query(BaseModel):
    question: str

@app.post("/predict/")
def predict(query: Query):
    prompt = f"### Question:\n{query.question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Answer:")[-1].strip()
    return {"answer": response}




# Weekly pipeline
new_logs = load_logs("user_logs.jsonl")  # Contains: {prompt, response, feedback}
filtered = [log for log in new_logs if log["feedback"] == "bad"]

with open("dataset/fine_tune_data.jsonl", "a") as f:
    for entry in filtered:
        json.dump({"prompt": entry["prompt"], "response": entry["corrected_response"]}, f)
        f.write("\n")

# Re-run fine-tuning every 2 weeks


# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from peft import get_peft_model, LoraConfig, TaskType
# import torch

# model_name = "mistralai/Mistral-7B-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token  # Fix padding

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_4bit=True,  # QLoRA style
#     device_map="auto"
# )

# # Apply LoRA config
# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     task_type=TaskType.CAUSAL_LM,
#     lora_dropout=0.05,
#     bias="none",
#     target_modules=["q_proj", "v_proj"]  # Depends on model
# )
# model = get_peft_model(model, peft_config)

# # Dataset: load your jsonl file
# from datasets import load_dataset
# dataset = load_dataset("json", data_files="dataset/fine_tune_data.jsonl")

# # Tokenization
# def tokenize(example):
#     return tokenizer(
#         example["prompt"],
#         text_target=example["response"],
#         truncation=True,
#         padding="max_length",
#         max_length=512
#     )

# tokenized_dataset = dataset.map(tokenize, batched=True)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./lora-mistral",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     lr_scheduler_type="cosine",
#     learning_rate=2e-4,
#     warmup_steps=100,
#     report_to="none"
# )

# # Train
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
# )

# trainer.train()




# !pip install transformers datasets peft accelerate bitsandbytes
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# import torch
# import json

# # Load base + LoRA model
# base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", load_in_4bit=True, device_map="auto")
# model = PeftModel.from_pretrained(base_model, "./lora-mistral")
# model.eval()

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# tokenizer.pad_token = tokenizer.eos_token


# def generate_response(prompt, max_tokens=150):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.7
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# from datasets import load_dataset

# test_data = load_dataset("json", data_files="dataset/test_set.jsonl")["train"]

# results = []
# for item in test_data:
#     prompt = item["prompt"]
#     reference = item["reference"]
#     prediction = generate_response(prompt)
#     results.append({"prompt": prompt, "prediction": prediction, "reference": reference})

# # Save results
# with open("dataset/evaluation_results.json", "w") as f:
#     json.dump(results, f, indent=2)

# pip install evaluate


# import evaluate

# rouge = evaluate.load("rouge")
# bleu = evaluate.load("bleu")

# preds = [r["prediction"] for r in results]
# refs = [r["reference"] for r in results]

# print("ROUGE:", rouge.compute(predictions=preds, references=refs))
# print("BLEU:", bleu.compute(predictions=preds, references=refs))

# import os
# from langchain.document_loaders import TextLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma

# def load_documents(directory: str):
#     """
#     Load all .txt and .md documents from the given directory.
#     """
#     loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True)
#     documents = loader.load()
#     print(f"[INFO] Loaded {len(documents)} documents from {directory}")
#     return documents

# def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
#     """
#     Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
#     """
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = splitter.split_documents(documents)
#     print(f"[INFO] Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
#     return chunks

# def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#     """
#     Generate embeddings for document chunks using Hugging Face model.
#     """
#     embedder = HuggingFaceEmbeddings(model_name=model_name)
#     print(f"[INFO] Using embedding model: {model_name}")
#     return embedder, chunks

# def store_in_chroma(chunks, embedder, persist_directory="chroma_db"):
#     """
#     Store chunks and embeddings in a local ChromaDB.
#     """
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedder,
#         persist_directory=persist_directory
#     )
#     vectorstore.persist()
#     print(f"[INFO] Stored embeddings in ChromaDB at: {persist_directory}")

# def main():
#     docs_path = "docs"  # Folder containing input .txt files
#     persist_path = "chroma_db"  # Where to store Chroma vector DB

#     print("=== Phase 4: RAG - Chunking + Embedding ===")
#     docs = load_documents(docs_path)
#     chunks = chunk_documents(docs)
#     embedder, processed_chunks = generate_embeddings(chunks)
#     store_in_chroma(processed_chunks, embedder, persist_path)

# if __name__ == "__main__":
#     main()



# import os
# from langchain.document_loaders import TextLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma

# DOCS_DIR = "docs"
# CHROMA_DIR = "chroma_db"
# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Small, fast & effective

# def load_and_chunk_docs(path: str, chunk_size: int = 500, overlap: int = 100):
#     print(f"[INFO] Loading documents from: {path}")
#     loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
#     documents = loader.load()

#     print(f"[INFO] Loaded {len(documents)} documents. Chunking...")
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
#     chunks = splitter.split_documents(documents)
#     print(f"[INFO] Generated {len(chunks)} chunks.")
#     return chunks

# def create_embeddings(model_name: str):
#     print(f"[INFO] Loading embedding model: {model_name}")
#     return HuggingFaceEmbeddings(model_name=model_name)

# def store_in_vector_db(chunks, embedder, persist_path: str):
#     print(f"[INFO] Storing embeddings in Chroma DB at: {persist_path}")
#     db = Chroma.from_documents(documents=chunks, embedding=embedder, persist_directory=persist_path)
#     db.persist()
#     print("[INFO] Vector DB storage complete.")

# def main():
#     chunks = load_and_chunk_docs(DOCS_DIR)
#     embedder = create_embeddings(EMBED_MODEL_NAME)
#     store_in_vector_db(chunks, embedder, CHROMA_DIR)

# if __name__ == "__main__":
#     main()



# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceHub
# from langchain.prompts import PromptTemplate

# # === CONFIGURATION ===
# CHROMA_DIR = "chroma_db"
# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.1"  # Or any Hugging Face-compatible model
# HF_TOKEN = "your_huggingface_api_token"  # 🔐 Required if model is gated

# def load_vectorstore():
#     embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
#     vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
#     print("[INFO] Loaded Chroma vector DB with retriever.")
#     return vectordb.as_retriever(search_kwargs={"k": 3})

# def load_llm():
#     llm = HuggingFaceHub(
#         repo_id=LLM_REPO_ID,
#         model_kwargs={"temperature": 0.7, "max_new_tokens": 512},
#         huggingfacehub_api_token=HF_TOKEN,
#     )
#     print(f"[INFO] Loaded LLM from Hugging Face: {LLM_REPO_ID}")
#     return llm

# def build_qa_chain(retriever, llm):
#     template = """
#     You are a helpful medical assistant. Use the following context to answer the user's question.
#     If you don't know the answer, just say you don't know. Don't make things up.

#     Context:
#     {context}

#     Question: {question}

#     Answer:"""

#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["context", "question"]
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",  # Or "map_reduce", "refine" for more control
#         retriever=retriever,
#         chain_type_kwargs={"prompt": prompt},
#         return_source_documents=True,
#     )

#     print("[INFO] RAG chain ready.")
#     return qa_chain

# def ask_question(qa_chain):
#     while True:
#         query = input("\n🧠 Ask your question (or type 'exit'): ").strip()
#         if query.lower() == "exit":
#             break
#         response = qa_chain(query)
#         print("\n📘 Answer:")
#         print(response["result"])

#         print("\n📄 Top source chunks used:")
#         for doc in response["source_documents"]:
#             print(f"- {doc.metadata.get('source', 'N/A')} | Chunk preview: {doc.page_content[:100]}...")

# if __name__ == "__main__":
#     retriever = load_vectorstore()
#     llm = load_llm()
#     qa_chain = build_qa_chain(retriever, llm)
#     ask_question(qa_chain)
