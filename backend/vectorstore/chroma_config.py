import os
from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=os.path.join(os.getcwd(), "vectorstorage", "chroma_db"),
    anonymized_telemetry=False
)
