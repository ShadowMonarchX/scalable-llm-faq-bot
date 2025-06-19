
# llm_logic/prompts.py

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
