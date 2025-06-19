from fastapi import APIRouter
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

router = APIRouter()

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("gpt-finetuned-customer-support")
model = AutoModelForCausalLM.from_pretrained("gpt-finetuned-customer-support").eval()

class Query(BaseModel):
    question: str

@router.post("/predict/")
def predict(query: Query):
    prompt = f"### Question:\n{query.question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Answer:")[-1].strip()
    return {"answer": response}
