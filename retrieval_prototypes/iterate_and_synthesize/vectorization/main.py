# vectorization/main.py
import os
import yaml
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    print("Vectorization component")
    config = get_config()
    vector_db_config = config["vector_db"]
    embedding_config = config["embeddings"]

    # Load raw documents
    with open("data_ingestion/raw_documents.pkl", "rb") as f:
        docs = pickle.load(f)

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="split_parents",
        persist_directory=vector_db_config["path"],
        embedding_function=OllamaEmbeddings(model=embedding_config["model"]),
    )
    
    # The storage layer for the parent documents
    store = InMemoryStore()

    # The retriever (and doc loader)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    print("Adding documents to the retriever...")
    retriever.add_documents(docs, ids=None)
    print("Documents added successfully.")

    # Save the docstore
    with open("data_ingestion/docstore.pkl", "wb") as f:
        pickle.dump(store, f)
    print("Docstore saved.")

    print(f"Vector store created at: {vector_db_config['path']}")

if __name__ == "__main__":
    main()
