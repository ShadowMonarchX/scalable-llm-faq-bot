from vector_store.vector_handler import VectorStoreHandler

def test_add_and_query_documents():
    handler = VectorStoreHandler()
    
    # Add a test doc
    test_doc = "The iPhone 14 Pro supports Dynamic Island."
    test_metadata = {"source": "unit_test"}
    test_id = "iphone-doc-1"
    
    handler.add_documents([test_doc], [test_metadata], [test_id])
    
    # Run a simple query
    results = handler.query("What is Dynamic Island?")
    
    assert isinstance(results, list)
    assert any("dynamic" in doc.lower() for doc in results)
