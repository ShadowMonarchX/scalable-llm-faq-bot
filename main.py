# main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import initialize_agent
from langchain_community.llms import HuggingFaceHub
from mcp_tools.retriever_tool import retriever_tool
from mcp_tools.llm_tool import llm_tool
from api import all_routes  # Combine and expose all routers

# ------------------ FastAPI Setup ------------------ #
app = FastAPI(
    title="Scalable LLM-Powered FAQ Bot",
    description="FastAPI + LangChain + Hugging Face + MCP Integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to allowed domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Register API Routes ------------------ #
for route in all_routes:
    app.include_router(route, prefix="/api")

# ------------------ LLM + Agent (LangChain) ------------------ #
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # Switch to your HF repo if needed
    model_kwargs={"temperature": 0.5, "max_new_tokens": 256}
)

tools = [retriever_tool, llm_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# ------------------ Health Check ------------------ #
@app.get("/")
async def root():
    return {"message": "Scalable FAQ Bot is running."}

# ------------------ MCP-Compatible Endpoint ------------------ #
@app.post("/mcp-query")
async def mcp_query(payload: dict):
    query = payload.get("question")
    if not query:
        return {"error": "Missing 'question' in payload."}
    try:
        response = agent.run(query)
        return {"response": response}
    except Exception as e:
        return {"error": f"Agent failed: {str(e)}"}
