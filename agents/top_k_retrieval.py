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
        self.llm_config = self.config["llm"]
        self.provider = self.llm_config.get("provider")
        self.response_generator = ResponseGenerator(prototype_name="Top-K Retrieval")
    
    def _setup_provider(self):
        """Setup LLM provider (WCA or Ollama)."""
        if self.provider == "wca":
            api_key = os.environ.get("WCA_API_KEY")
            if not api_key:
                raise ValueError("WCA_API_KEY not found in environment variables.")
            return WCAService(api_key=api_key)
        else:
            return ChatOllama(
                model=self.llm_config["ollama_model"],
                temperature=0,
                top_k=40,
                top_p=0.9
            )
    
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

    def _create_qa_chain(self, retriever):
        """Create the QA chain for the agent."""
        if self.provider == "wca":
            # For WCA, we'll handle this differently in the run method
            return retriever
        else:
            # For Ollama, create a proper chain with dynamic prompting
            llm = self._setup_provider()
            
            # We'll create the prompt dynamically in the run method
            # For now, create a placeholder that will be replaced
            template = """You are a senior software architect. Based on the following set of retrieved source files, produce a comprehensive analysis of the project that directly answers the user's question.

Retrieved Documents (Context):
{context}

User's Question:
{question}

Architectural Analysis:"""
            
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            
            # Create document chain
            combine_docs_chain = StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=prompt),
                document_variable_name="context"
            )
            
            # Create question generator
            question_generator = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    template="Given the following conversation and a follow up question, "
                    "rephrase the follow up question to be a standalone question.\n\n"
                    "Chat History:\n{chat_history}\n"
                    "Follow Up Input: {question}\n"
                    "Standalone question:",
                    input_variables=["chat_history", "question"]
                )
            )
            
            # Create conversational retrieval chain
            qa = ConversationalRetrievalChain(
                retriever=retriever,
                combine_docs_chain=combine_docs_chain,
                question_generator=question_generator,
                return_source_documents=True,
            )
            
            return qa
    
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
        top_k = self.config["retrieval"]["top_k"]
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