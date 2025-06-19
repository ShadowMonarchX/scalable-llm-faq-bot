# api/model_api.py

from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

router = APIRouter()

# Load model and tokenizer once at startup
MODEL_PATH = "gpt-finetuned-customer-support"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
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
