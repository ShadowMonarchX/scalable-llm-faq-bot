from chromadb.config import Settings

# ChromaDB in-memory or persistent setup
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",  # persistent storage
    anonymized_telemetry=False
)
