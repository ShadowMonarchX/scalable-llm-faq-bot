from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

router = APIRouter()

# -------- Load Model -------- #
MODEL_PATH = "./models/final"  # Your saved model path
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True).eval().to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load local model: {e}")

# -------- Pydantic Input -------- #
class Query(BaseModel):
    question: str

# -------- Inference Endpoint -------- #
@router.post("/predict")
async def predict(query: Query):
    prompt = f"### Question:\n{query.question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
