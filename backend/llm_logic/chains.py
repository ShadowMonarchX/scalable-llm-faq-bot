from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from llm_logic.prompts import FAQ_PROMPT


def build_chain(llm: HuggingFacePipeline, retriever) -> RunnableMap:
    """
    Builds a LangChain chain using a Hugging Face LLM and retriever (e.g., ChromaDB retriever).
    Returns a RunnableMap that can be used to run the full QA pipeline.
    """
    chain = RunnableMap({
        "context": retriever,
        "question": RunnablePassthrough(),
    }) | FAQ_PROMPT | llm

    return chain
