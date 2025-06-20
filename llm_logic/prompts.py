# # llm_logic/prompts.py

from langchain.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful customer support assistant for a smartphone company. Use the provided context to answer the user's question clearly and concisely.

### Context:
{context}

### Question:
{question}

### Answer:
""",
)


# # llm_logic/prompts.py

# import os
# import psutil
# import platform
# import torch
# import GPUtil
# from dotenv import load_dotenv
# from transformers import pipeline

# def get_system_info():
#     print("\nSystem Information:")
#     print("----------------------")
    
#     # CPU
#     print(f"CPU: {platform.processor()}")
#     print(f"CPU Cores: {os.cpu_count()}")

#     # RAM
#     total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
#     print(f"RAM: {total_ram_gb} GB")

#     # GPU
#     if torch.cuda.is_available():
#         gpus = GPUtil.getGPUs()
#         for gpu in gpus:
#             print(f"GPU: {gpu.name} | {gpu.memoryTotal} MB VRAM")
#     else:
#         print("GPU: Not available (using CPU only)")

#     # Basic Capability Checks
#     print("\nCapability Check:")
#     print("----------------------")
#     if total_ram_gb < 8:
#         print("Warning: Less than 8GB RAM may cause memory issues with LLMs.")
#     else:
#         print("RAM is sufficient.")

#     if torch.cuda.is_available():
#         print("GPU is available for acceleration.")
#     else:
#         print(" GPU not detected. Using CPU only.")

# # Run system diagnostics
# get_system_info()

# # Load environment
# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")  # Optional for open models

# # Load the LLM pipeline
# print("\nLoading model: tiiuae/falcon-rw-1b...")
# pipe = pipeline(
#     "text-generation",
#     model="tiiuae/falcon-rw-1b",
#     device="cpu"
# )

# # Run a query
# query = "What are the symptoms of pneumonia?"
# print(f"\nQuery: {query}")
# response = pipe(query, max_new_tokens=100)
# print(f"\nResponse: {response[0]['generated_text']}")
