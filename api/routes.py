from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from vector_store.vector_handler import VectorStoreHandler
from llm_logic.chains import generate_answer

router = APIRouter()
vector_store = VectorStoreHandler()

class QueryRequest(BaseModel):
    question: str

class UploadRequest(BaseModel):
    documents: list[str]
    metadatas: list[dict]
    ids: list[str]

@router.get("/health")
async def health_check():
    return {"status": "✅ Server is running."}

@router.post("/query")
async def query_handler(request: QueryRequest):
    try:
        retrieved_docs = vector_store.query(request.question)
        response = generate_answer(request.question, retrieved_docs)
        return {"question": request.question, "response": response, "retrieved": retrieved_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_documents(request: UploadRequest):
    try:
        vector_store.add_documents(request.documents, request.metadatas, request.ids)
        return {"status": "✅ Documents uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
