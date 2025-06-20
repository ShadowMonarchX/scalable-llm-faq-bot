# vector_store/chroma_config.py

from chromadb.config import Settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
    anonymized_telemetry=False
)
client = chromadb.Client(settings=CHROMA_SETTINGS)  # ← THIS is outdated
