# api/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from vector_store.vector_handler import VectorStoreHandler
from llm_logic.chains import build_qa_chain

router = APIRouter()
qa_chain = build_qa_chain()

class QueryRequest(BaseModel):
    question: str

class UploadRequest(BaseModel):
    documents: list[str]
    metadatas: list[dict]
    ids: list[str]

@router.get("/health")
async def health_check():
    return {"status": "Server is running ✅"}

@router.post("/query")
async def query_handler(request: QueryRequest):
    try:
        response = qa_chain.run(request.question)
        return {"question": request.question, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.post("/upload")
async def upload_documents(request: UploadRequest):
    try:
        vector_store = VectorStoreHandler()
        vector_store.add_documents(request.documents, request.metadatas, request.ids)
        return {"status": "Documents uploaded successfully ✅"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
