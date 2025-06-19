import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

# Load fine-tuned model
model_path = "./gpt-finetuned-customer-support"  # Update if model path differs
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Response generator
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
val_path = "dataset/val.jsonl"  # Or use dataset/reasonmed/val.jsonl
assert os.path.exists(val_path), f"Validation file not found: {val_path}"

val_samples = []
with open(val_path, 'r', encoding='utf-8') as f:
    for line in f:
        val_samples.append(json.loads(line))

# Generate predictions
results = []
for sample in val_samples[:10]:  # Limit to 10 for preview
    prompt = sample['prompt']
    expected = sample['response']
    predicted = generate_response(prompt)
    results.append((prompt, expected, predicted))
    print(f"\n🟡 Question: {prompt}")
    print(f"✅ Expected: {expected}")
    print(f"🔷 Predicted: {predicted}")

# ROUGE-L Evaluation
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = []

for _, expected, predicted in results:
    score = scorer.score(expected, predicted)
    scores.append(score['rougeL'].fmeasure)

avg_rouge = sum(scores) / len(scores)
print(f"\n📊 Average ROUGE-L F1 Score: {avg_rouge:.4f}")

# Custom Query
custom_query = "How do I change my phone's ringtone?"
custom_response = generate_response(custom_query)
print(f"\n📨 Custom Query: {custom_query}")
print(f"🤖 Model Response: {custom_response}")
