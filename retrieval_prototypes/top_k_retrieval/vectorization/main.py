# vectorization/main.py
import os
import yaml
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

def main():
    print("Vectorization component")
    
    config_path = "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Root configuration file not found at {config_path}")
        sys.exit(1)

    storage_config = config["storage"]
    embedding_config = config["embeddings"]

    # Load raw documents from the central path
    raw_docs_path = storage_config["raw_docs"]
    try:
        with open(raw_docs_path, "rb") as f:
            docs = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {raw_docs_path} not found. Please run the central data_ingestion.py script first.")
        return

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    
    vectorstore = Chroma(
        collection_name="split_parents",
        persist_directory=storage_config["vector_store"],
        embedding_function=OllamaEmbeddings(model=embedding_config["model"]),
    )
    
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    print("Adding documents to the retriever...")
    retriever.add_documents(docs, ids=None)
    print("Documents added successfully.")

    docstore_path = storage_config["doc_store"]
    with open(docstore_path, "wb") as f:
        pickle.dump(store, f)
    print(f"Docstore saved to: {docstore_path}")

    print(f"Vector store created at: {storage_config['vector_store']}")

if __name__ == "__main__":
    main()