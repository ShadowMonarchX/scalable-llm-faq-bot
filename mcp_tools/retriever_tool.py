from langchain.tools import Tool
from vector_store.vector_handler import VectorStoreHandler

# Initialize your custom vector store
vector_store = VectorStoreHandler()

def retrieve_docs_tool(query: str) -> str:
    """
    Retrieves top documents relevant to the query using ChromaDB.
    """
    results = vector_store.query(query, top_k=3)
    return "\n---\n".join(results)

retriever_tool = Tool(
    name="RetrieverTool",
    func=retrieve_docs_tool,
    description="Useful for retrieving relevant documents from the FAQ vector database. Input should be a user question."
)
