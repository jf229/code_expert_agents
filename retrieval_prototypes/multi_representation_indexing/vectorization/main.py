# vectorization/main.py
import os
import yaml
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
import uuid
import sys

def main():
    print("--- Vectorization (Multi-Representation) ---")
    
    config_path = "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Root configuration file not found at {config_path}")
        sys.exit(1)

    storage_config = config["storage"]
    embedding_config = config["embeddings"]

    representations_path = storage_config["multi_representations"]
    try:
        with open(representations_path, "rb") as f:
            representations = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {representations_path} not found.")
        print("Please run the representation_builder.py script first.")
        return

    vectorstore = Chroma(
        collection_name="multi_representation",
        persist_directory=storage_config["vector_store"],
        embedding_function=OllamaEmbeddings(model=embedding_config["model"]),
    )
    
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # This is the correct way to add documents to the MultiVectorRetriever
    doc_ids = [str(uuid.uuid4()) for _ in representations]
    
    # Create the documents to be indexed (summaries and questions)
    index_docs = []
    for i, rep in enumerate(representations):
        # Add the summary
        index_docs.append(Document(page_content=rep["summary"], metadata={id_key: doc_ids[i], 'source': rep["source"]}))
        # Add the hypothetical questions
        for q in rep["hypothetical_questions"]:
            if q:
                index_docs.append(Document(page_content=q, metadata={id_key: doc_ids[i], 'source': rep["source"]}))

    # Create the documents to be stored (the original content)
    store_docs = [
        Document(page_content=rep["original_content"], metadata={id_key: doc_ids[i], 'source': rep["source"]})
        for i, rep in enumerate(representations)
    ]

    print("Adding documents to the retriever...")
    retriever.vectorstore.add_documents(index_docs)
    retriever.docstore.mset(list(zip(doc_ids, store_docs)))
    print("Documents added successfully.")

    docstore_path = storage_config["doc_store"]
    with open(docstore_path, "wb") as f:
        pickle.dump(store, f)
    print(f"Docstore saved to: {docstore_path}")

    print(f"Vector store created at: {storage_config['vector_store']}")

if __name__ == "__main__":
    main()
