from fastapi import APIRouter
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

router = APIRouter()

# Load fine-tuned model from local directory
model_path = "gpt-finetuned-customer-support"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).eval()

class Query(BaseModel):
    question: str

@router.post("/predict")
def predict(query: Query):
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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Answer:")[-1].strip()
    return {"answer": response}
