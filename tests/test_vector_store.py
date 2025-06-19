from vector_store.vector_handler import VectorStoreHandler

def test_add_and_query_documents():
    handler = VectorStoreHandler()

    # Add sample doc
    doc = "The iPhone 14 Pro supports Dynamic Island."
    metadata = {"source": "unit_test"}
    doc_id = "iphone-doc-1"

    handler.add_documents([doc], [metadata], [doc_id])

    # Query
    results = handler.query("What is Dynamic Island?")

    assert isinstance(results, list)
    assert any("dynamic" in doc.lower() for doc in results)
