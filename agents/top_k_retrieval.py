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
from shared.base_agent import BaseAgent
from shared.storage_manager import get_project_file_path, ensure_project_storage


class TopKVectorizer:
    """Handles vectorization for Top-K retrieval strategy."""

    def __init__(self, config, repo_path: str = None):
        self.config = config
        self.storage_config = config["storage"]
        self.embedding_config = config["embeddings"]
        self.retrieval_config = config["retrieval"]
        self.llm_config = config["llm"]
        self.repo_path = repo_path

    def create_retriever(self):
        """Create ParentDocumentRetriever with text splitting."""
        print("Creating Top-K vectorization...")

        # Load documents
        raw_docs_path = self.storage_config["raw_docs"]
        with open(raw_docs_path, "rb") as f:
            docs = pickle.load(f)

        # Create text splitter for child documents
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

        # Get vector store path (use project storage if repo_path provided)
        if self.repo_path:
            vector_store_path = get_project_file_path(self.repo_path, "vector_store")
        else:
            vector_store_path = self.storage_config["vector_store"]

        # Create vectorstore
        embedding_endpoint = self.llm_config.get("ollama_endpoint", "http://localhost:11434")
        vectorstore = Chroma(
            collection_name="split_parents",
            persist_directory=vector_store_path,
            embedding_function=OllamaEmbeddings(
                model=self.embedding_config["model"],
                base_url=embedding_endpoint
            ),
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


class TopKAgent(BaseAgent):
    """Main Top-K Retrieval Agent using BaseAgent template."""

    def __init__(self, repo_path: str = None):
        load_dotenv()
        super().__init__(name="Top-K Retrieval")
        self.repo_path = repo_path or os.environ.get("REPO_PATH")
        if self.repo_path:
            ensure_project_storage(self.repo_path)

    def _classify_question_type(self, question):
        """Classify question to use appropriate prompt strategy."""
        question_lower = question.lower()

        if any(word in question_lower for word in ['architecture', 'overview', 'structure', 'design', 'pattern']):
            return 'architectural'
        elif any(word in question_lower for word in ['how', 'implement', 'work', 'process', 'algorithm']):
            return 'implementation'
        elif any(word in question_lower for word in ['what', 'class', 'function', 'method', 'specific']):
            return 'entity_specific'
        elif any(word in question_lower for word in ['why', 'reason', 'purpose', 'decision', 'rationale']):
            return 'rationale'
        elif any(word in question_lower for word in ['flow', 'data', 'process', 'workflow', 'pipeline']):
            return 'data_flow'
        else:
            return 'general'

    def _get_dynamic_prompt_template(self, question_type):
        """Get prompt template optimized for question type."""

        templates = {
            'architectural': """You are a senior software architect analyzing system design. Focus on architectural aspects of this codebase.

Based on the retrieved source files, analyze the system architecture with focus on:
- Overall system design and architectural patterns
- Component relationships and dependencies
- Design decisions and trade-offs
- System boundaries and interfaces

Retrieved Documents (Context):
{context}

User's Question: {question}

Provide an architectural analysis with these sections:
1. **Architectural Overview:** High-level system design
2. **Key Components:** Main architectural components and responsibilities
3. **Design Patterns:** Architectural patterns used
4. **Component Relationships:** How components interact
5. **Design Rationale:** Why this architecture was chosen

Analysis:""",

            'implementation': """You are a senior developer explaining code implementation. Focus on HOW things work in this codebase.

Based on the retrieved source files, explain the implementation with focus on:
- Step-by-step process flows and algorithms
- Key implementation details and mechanisms
- Code execution paths and control flow
- Data transformations and processing logic

Retrieved Documents (Context):
{context}

User's Question: {question}

Provide an implementation analysis with these sections:
1. **Process Overview:** High-level explanation of how it works
2. **Implementation Steps:** Detailed step-by-step breakdown
3. **Key Algorithms:** Important algorithms and logic
4. **Code Flow:** How execution flows through components
5. **Technical Details:** Important implementation specifics

Analysis:""",

            'entity_specific': """You are a code expert explaining specific code entities. Focus on particular classes, functions, or components.

Based on the retrieved source files, explain the specific entity with focus on:
- Purpose and responsibilities of the entity
- Input/output parameters and return types
- Internal logic and behavior
- Usage patterns and integration points

Retrieved Documents (Context):
{context}

User's Question: {question}

Provide an entity analysis with these sections:
1. **Entity Purpose:** What this entity does and why it exists
2. **Interface Definition:** Parameters, return types, public methods
3. **Internal Logic:** How it works internally
4. **Usage Context:** How and where it's used
5. **Dependencies:** What it depends on and what depends on it

Analysis:""",

            'general': """You are a senior software architect. Based on the retrieved source files, provide a comprehensive analysis that directly answers the user's question.

Retrieved Documents (Context):
{context}

User's Question: {question}

Provide a comprehensive analysis with:
1. **Direct Answer:** Address the question directly
2. **Core Components:** Key components relevant to the question
3. **Technical Details:** Important implementation details
4. **Context:** How this fits into the broader system
5. **Additional Insights:** Other relevant information

Analysis:"""
        }

        return templates.get(question_type, templates['general'])

    def prepare(self, question: str) -> None:
        # Ensure raw documents exist by running central ingestion
        print("--- Data Ingestion ---")
        data_ingestion_main()

    def retrieve(self, question: str):
        print("--- Vectorization ---")
        vectorizer = TopKVectorizer(self.config, repo_path=self.repo_path)
        retriever = vectorizer.create_retriever()
        print("--- Retrieval ---")
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} documents.")
        return docs


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
    repo_path = args.repo or os.environ.get("REPO_PATH")
    if args.repo:
        os.environ["REPO_PATH"] = args.repo

    # Run the agent
    agent = TopKAgent(repo_path=repo_path)
    agent.run(args.question)


if __name__ == "__main__":
    main()
