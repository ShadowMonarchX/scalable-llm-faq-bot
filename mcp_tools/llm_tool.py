from langchain.tools import Tool
from llm_logic.chains import generate_answer

def answer_query_tool(query: str) -> str:
    """
    Generates an answer to the user query using the LLM and relevant docs.
    """
    # You could enhance this to use actual retrieved context
    return generate_answer(query)

llm_tool = Tool(
    name="LLMAnswerTool",
    func=answer_query_tool,
    description="Use this to answer customer questions based on vector search context."
)
