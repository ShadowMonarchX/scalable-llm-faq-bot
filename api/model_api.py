from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Load model and tokenizer once at startup
MODEL_PATH = "gpt-finetuned-customer-support"
HF_TOKEN = os.getenv("HF_TOKEN") # Add this to your .env file

try:
    if HF_TOKEN:
        login(token=HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, token=HF_TOKEN).eval()
    model.to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

class Query(BaseModel):
    question: str

@router.post("/predict")
async def predict(query: Query):
    prompt = f"### Question:\n{query.question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("### Answer:")[-1].strip()
    return {"answer": answer}
