#!/usr/bin/env python3
"""
Top-K Retrieval RAG Agent

This is the baseline approach for RAG-based code analysis. It's fast and effective 
for clear, specific questions. It performs data ingestion and vectorization automatically.

Usage:
    python top_k_retrieval.py "Provide an architectural analysis of the StravaService"
    python top_k_retrieval.py "How does authentication work?" --repo /path/to/repo

Use Case: Getting a quick, high-level analysis based on the most relevant files.
"""

import argparse
import os
import pickle
import yaml
from dotenv import load_dotenv

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import shared services and central data ingestion
from shared import WCAService, ResponseGenerator, load_config, setup_and_pull_models, data_ingestion_main


class TopKVectorizer:
    """Handles vectorization for Top-K retrieval strategy."""
    
    def __init__(self, config):
        self.config = config
        self.storage_config = config["storage"]
        self.embedding_config = config["embeddings"]
        self.retrieval_config = config["retrieval"]
    
    def create_retriever(self):
        """Create ParentDocumentRetriever with text splitting."""
        print("Creating Top-K vectorization...")
        
        # Load documents
        raw_docs_path = self.storage_config["raw_docs"]
        with open(raw_docs_path, "rb") as f:
            docs = pickle.load(f)
        
        # Create text splitter for child documents
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        
        # Create vectorstore
        vectorstore = Chroma(
            collection_name="split_parents",
            persist_directory=self.storage_config["vector_store"],
            embedding_function=OllamaEmbeddings(model=self.embedding_config["model"]),
        )
        
        # Create storage for parent documents
        store = InMemoryStore()
        
        # Create retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
        
        # Configure search kwargs for top-k
        retriever.search_kwargs = {"k": self.retrieval_config["top_k"]}
        
        print("Adding documents to the retriever...")
        retriever.add_documents(docs, ids=None)
        print("Documents added successfully.")
        
        # Save docstore
        docstore_path = self.storage_config["doc_store"]
        with open(docstore_path, "wb") as f:
            pickle.dump(store, f)
        print(f"Docstore saved to: {docstore_path}")
        
        print(f"Vector store created at: {self.storage_config['vector_store']}")
        
        return retriever


class TopKAgent:
    """Main Top-K Retrieval Agent."""
    
    def __init__(self):
        load_dotenv()
        self.config = load_config()
    
    def run(self, question):
        """Run the Top-K retrieval agent with the given question."""
        print("--- RAG-based Code Expert Agent (Top-K Retrieval) ---")
        
        # Step 1: Data Ingestion
        print("--- Data Ingestion ---")
        data_ingestion_main()
        
        # Step 2: Vectorization
        print("--- Vectorization ---")
        vectorizer = TopKVectorizer(self.config)
        retriever = vectorizer.create_retriever()
        
        # Step 3: Agent Orchestration
        print("--- Agent Orchestration ---")
        
        # Get top-k documents
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} documents.")
        
        if not docs:
            print("No relevant documents found for the query.")
            return
        
        # Use unified response generator for any provider
        from shared import UnifiedResponseGenerator
        unified_generator = UnifiedResponseGenerator(self.config, prototype_name="Top-K Retrieval")
        unified_generator.generate_response_with_context(question, docs)
        
        print("--- System Finished ---")


def main():
    """Main entry point for Top-K Retrieval prototype."""
    parser = argparse.ArgumentParser(description="RAG-based Code Expert Agent (Top-K Retrieval)")
    parser.add_argument("question", type=str, help="The question to ask the agent.")
    parser.add_argument("--repo", help="Repository path to analyze")
    parser.add_argument("--privacy", action="store_true", help="Enable privacy mode for testing")
    parser.add_argument("--no-privacy", action="store_true", help="Disable privacy mode for testing")
    args = parser.parse_args()
    
    # Load configuration and pull models if needed
    config = load_config()
    
    # Override privacy setting based on command line args
    if args.privacy and args.no_privacy:
        print("Error: Cannot specify both --privacy and --no-privacy")
        return
    elif args.privacy:
        config["privacy"]["enable"] = True
        print("Privacy mode: ENABLED for testing")
    elif args.no_privacy:
        config["privacy"]["enable"] = False
        print("Privacy mode: DISABLED for testing")
    else:
        privacy_status = "ENABLED" if config["privacy"]["enable"] else "DISABLED"
        print(f"Privacy mode: {privacy_status} (default)")
    
    # Setup and pull required models
    setup_and_pull_models(config)
    
    # Set repo path for data ingestion
    if args.repo:
        os.environ["REPO_PATH"] = args.repo
    
    # Run the agent
    agent = TopKAgent()
    agent.run(args.question)


if __name__ == "__main__":
    main()