# mcp_tools/llm_tool.py

from langchain.tools import Tool
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",
    tokenizer="EleutherAI/gpt-neo-125M",
    max_new_tokens=100
)

def simple_llm_response(question: str) -> str:
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    result = pipe(prompt, return_full_text=False)[0]["generated_text"]
    return result.strip()

llm_tool = Tool(
    name="LLMTool",
    func=simple_llm_response,
    description="Use this tool to directly generate answers from the base LLM without retrieval."
)
