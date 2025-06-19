from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import initialize_agent
from langchain.llms import HuggingFaceHub
from mcp_tools.retriever_tool import retriever_tool
from mcp_tools.llm_tool import llm_tool
from api import all_routes  # Combines routes from routes.py and model_api.py

# Initialize FastAPI app
app = FastAPI(
    title="Scalable LLM-Powered FAQ Bot",
    description="FastAPI + LangChain + HuggingFace + MCP integration",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routers (query, upload, health, predict)
for route in all_routes:
    app.include_router(route, prefix="/api")

# Load LLM from Hugging Face Hub (can switch to local model)
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # Or your own HF model
    model_kwargs={"temperature": 0.5, "max_new_tokens": 256}
)

# Initialize LangChain agent with MCP-compatible tools
tools = [retriever_tool, llm_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Root health endpoint
@app.get("/")
async def root():
    return {"message": "✅ Scalable FAQ Bot is running."}

# MCP-compatible agent query endpoint
@app.post("/mcp-query")
async def mcp_query(payload: dict):
    query = payload.get("question")
    if not query:
        return {"error": "Missing 'question' in payload."}
    response = agent.run(query)
    return {"response": response}
