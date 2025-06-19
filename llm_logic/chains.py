# llm_logic/chains.py

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from llm_logic.prompts import QA_PROMPT
from vector_store.vector_handler import load_vector_store
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def get_hf_llm(model_id: str = "EleutherAI/gpt-neo-125M") -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=pipe)


def build_qa_chain(persist_directory: str = "chroma_db", model_id: str = "EleutherAI/gpt-neo-125M") -> RetrievalQA:
    vectordb = load_vector_store(persist_directory)
    llm = get_hf_llm(model_id)

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    return qa_chain
