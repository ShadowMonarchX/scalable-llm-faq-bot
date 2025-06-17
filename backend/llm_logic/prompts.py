from langchain_core.prompts import PromptTemplate

# Prompt for answering a medical question based on retrieved context
FAQ_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and knowledgeable medical assistant.

Use the following context to answer the user's question as accurately and concisely as possible.
If the answer cannot be found in the context, reply: "Sorry, I don't have information on that."

Context:
{context}

Question:
{question}

Answer:
"""
)
