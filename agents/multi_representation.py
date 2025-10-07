#!/usr/bin/env python3
"""
Multi-Representation Indexing RAG Agent

A hybrid agent that uses different strategies for different types of questions.
Creates multiple representations of each document (summaries and hypothetical 
questions) and uses different retrieval strategies based on query type.

Usage:
    # Step 1: Build the multi-representations (run once)
    python multi_representation.py --build-representations
    
    # Step 2: Run queries with different strategies
    python multi_representation.py "explain this repo" --strategy broad
    python multi_representation.py "how is user authentication handled" --strategy specific

Use Case: Flexible analysis that adapts strategy based on question type.
"""

import argparse
import os
import pickle
import yaml
import uuid
from dotenv import load_dotenv
from tqdm import tqdm

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
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema.document import Document

# Import shared services and central data ingestion
from shared import WCAService, ResponseGenerator, load_config, setup_and_pull_models, data_ingestion_main
from shared.llm_providers import get_llm_provider



class RepresentationBuilder:
    """Builds multiple representations of documents for enhanced retrieval."""
    
    def __init__(self, config):
        load_dotenv()
        self.config = config
        self.llm_config = config["llm"]
        self.storage_config = config["storage"]
        self.provider = self.llm_config.get("provider")
    
    def _setup_provider(self):
        """Setup LLM provider (WCA or Ollama)."""
        if self.provider == "wca":
            api_key = os.environ.get("WCA_API_KEY")
            if not api_key:
                raise ValueError("WCA_API_KEY not found in environment variables.")
            return WCAService(api_key=api_key)
        else:
            # Use ChatOllama with correct config path
            model = self.llm_config["models"][self.provider]
            return ChatOllama(
                model=model,
                temperature=0,
                top_k=40,
                top_p=0.9
            )
    
    def _create_ollama_chains(self, llm):
        """Create LangChain chains for Ollama."""
        # Summary chain
        summary_template = """Provide a concise, one-paragraph summary of the following file's purpose and key components:

{file_content}

Summary:"""
        summary_prompt = PromptTemplate(template=summary_template, input_variables=["file_content"])
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        
        # Questions chain
        questions_template = """Based on the content of the following file, generate a list of 3-5 high-level questions that this file could help answer. Return only the questions, each on a new line.

{file_content}

Questions:"""
        questions_prompt = PromptTemplate(template=questions_template, input_variables=["file_content"])
        questions_chain = LLMChain(llm=llm, prompt=questions_prompt)
        
        return summary_chain, questions_chain
    
    def build_representations(self):
        """Build multiple representations for all documents."""
        print("--- Multi-Representation Builder ---")
        
        # First, ensure we have raw documents
        print("--- Data Ingestion ---")
        data_ingestion_main()
        
        # Load raw documents
        raw_docs_path = self.storage_config["raw_docs"]
        try:
            with open(raw_docs_path, "rb") as f:
                raw_docs = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: {raw_docs_path} not found.")
            print("Please run data ingestion first.")
            return False
        
        # Setup provider
        if self.provider == "wca":
            wca_service = self._setup_provider()
            summary_chain = None
            questions_chain = None
        else:
            llm = self._setup_provider()
            summary_chain, questions_chain = self._create_ollama_chains(llm)
            wca_service = None
        
        representations = []
        print(f"Generating summaries and questions for {len(raw_docs)} documents using provider: {self.provider}...")
        
        for doc in tqdm(raw_docs, desc="Generating Representations"):
            file_path = doc.metadata['source']
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if self.provider == "wca":
                    # Use WCA service and normalize output
                    summary_resp = wca_service.summarize_file(file_path, content)
                    summary = wca_service.extract_generated_text(summary_resp) or "No summary available"

                    questions_resp = wca_service.generate_hypothetical_questions(file_path, content)
                    questions_text = wca_service.extract_generated_text(questions_resp) or ""
                    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                else:
                    # Use Ollama
                    summary = summary_chain.run(file_content=content)
                    questions_text = questions_chain.run(file_content=content)
                    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                
                representations.append({
                    "source": file_path,
                    "original_content": content,
                    "summary": summary,
                    "hypothetical_questions": questions
                })
                
            except Exception as e:
                print(f"\nCould not process {file_path}: {e}")
                continue
        
        # Save representations
        output_path = self.storage_config["multi_representations"]
        with open(output_path, "wb") as f:
            pickle.dump(representations, f)
        
        print(f"\nSuccessfully generated and saved representations for {len(representations)} documents.")
        print(f"Representations saved to: {output_path}")
        
        return True


class MultiRepresentationVectorizer:
    """Creates vectorized multi-representation index."""
    
    def __init__(self, config):
        self.config = config
        self.storage_config = config["storage"]
        self.embedding_config = config["embeddings"]
    
    def create_retriever(self):
        """Create MultiVectorRetriever for multi-representation indexing."""
        print("Creating multi-representation vectorization...")
        
        # Load multi-representation documents
        multi_repr_path = self.storage_config["multi_representations"]
        try:
            with open(multi_repr_path, "rb") as f:
                representations = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: {multi_repr_path} not found.")
            print("Please run with --build-representations first.")
            return None
        
        # Create vectorstore
        vectorstore = Chroma(
            collection_name="multi_representation",
            persist_directory=self.storage_config["vector_store"],
            embedding_function=OllamaEmbeddings(model=self.embedding_config["model"]),
        )
        
        # Create storage for original documents
        store = InMemoryStore()
        id_key = "doc_id"
        
        # Create MultiVectorRetriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )
        
        print("Adding documents with multiple representations...")
        
        # Generate unique IDs for each document
        doc_ids = [str(uuid.uuid4()) for _ in representations]
        
        # Create the documents to be indexed (summaries and questions)
        index_docs = []
        for i, rep in enumerate(representations):
            # Add the summary
            index_docs.append(Document(
                page_content=rep["summary"], 
                metadata={id_key: doc_ids[i], 'source': rep["source"]}
            ))
            # Add the hypothetical questions
            for q in rep["hypothetical_questions"]:
                if q:
                    index_docs.append(Document(
                        page_content=q, 
                        metadata={id_key: doc_ids[i], 'source': rep["source"]}
                    ))
        
        # Create the documents to be stored (the original content)
        store_docs = [
            Document(
                page_content=rep["original_content"], 
                metadata={id_key: doc_ids[i], 'source': rep["source"]}
            )
            for i, rep in enumerate(representations)
        ]
        
        # Add documents to retriever
        retriever.vectorstore.add_documents(index_docs)
        retriever.docstore.mset(list(zip(doc_ids, store_docs)))
        print("Documents added successfully.")
        
        # Save docstore
        docstore_path = self.storage_config["doc_store"]
        with open(docstore_path, "wb") as f:
            pickle.dump(store, f)
        print(f"Docstore saved to: {docstore_path}")
        
        print(f"Vector store created at: {self.storage_config['vector_store']}")
        
        return retriever


class MultiRepresentationAgent:
    """Main Multi-Representation Agent."""
    
    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self.llm_config = self.config["llm"]
        self.storage_config = self.config["storage"]
        self.retrieval_config = self.config["retrieval"]
        self.provider = self.llm_config.get("provider")
        self.response_generator = ResponseGenerator(prototype_name="Multi-Representation Indexing")
    
    def _setup_provider(self):
        """Setup LLM provider (WCA or Ollama)."""
        if self.provider == "wca":
            api_key = os.environ.get("WCA_API_KEY")
            if not api_key:
                raise ValueError("WCA_API_KEY not found in environment variables.")
            return WCAService(api_key=api_key)
        else:
            # Use ChatOllama with correct config path
            model = self.llm_config["models"][self.provider]
            return ChatOllama(
                model=model,
                temperature=0,
                top_k=40,
                top_p=0.9
            )
    
    def _create_qa_chain(self, retriever):
        """Create the QA chain for the agent."""
        if self.provider == "wca":
            # For WCA, we'll handle this differently in the run method
            return retriever
        else:
            # For Ollama, create a proper chain
            llm = self._setup_provider()
            
            # Create prompt template
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
    
    def build_representations(self):
        """Build multi-representations."""
        builder = RepresentationBuilder(self.config)
        return builder.build_representations()
    
    def _analyze_question_features(self, question):
        """Analyze question to determine optimal strategy."""
        question_lower = question.lower()
        
        features = {
            'scope': 'general',
            'complexity': 'medium',
            'specificity': 'medium'
        }
        
        # Determine scope
        if any(word in question_lower for word in ['entire', 'whole', 'all', 'overall', 'complete', 'full']):
            features['scope'] = 'system_wide'
        elif any(word in question_lower for word in ['specific', 'particular', 'individual', 'single']):
            features['scope'] = 'specific_entity'
        elif any(word in question_lower for word in ['class', 'function', 'method', 'component']):
            features['scope'] = 'component_focused'
        
        # Determine complexity
        if any(word in question_lower for word in ['how', 'why', 'explain', 'detailed', 'deep', 'comprehensive']):
            features['complexity'] = 'high'
        elif any(word in question_lower for word in ['what', 'which', 'where', 'simple', 'basic']):
            features['complexity'] = 'low'
        
        # Determine specificity
        if any(word in question_lower for word in ['overview', 'summary', 'general', 'broad']):
            features['specificity'] = 'low'
        elif any(word in question_lower for word in ['detailed', 'specific', 'exact', 'precise']):
            features['specificity'] = 'high'
        
        return features
    
    def _select_optimal_strategy(self, question, user_strategy):
        """Select optimal retrieval strategy based on question analysis."""
        
        # Analyze question features
        features = self._analyze_question_features(question)
        
        # Define strategy configurations
        strategies = {
            'focused': {
                'name': 'focused',
                'k': 3,
                'description': 'Focused search for specific entities'
            },
            'broad': {
                'name': 'broad', 
                'k': 12,
                'description': 'Broad search for system-wide understanding'
            },
            'hybrid': {
                'name': 'hybrid',
                'k': 8, 
                'description': 'Balanced approach for moderate complexity'
            },
            'deep_dive': {
                'name': 'deep_dive',
                'k': 6,
                'description': 'Deep analysis for complex questions'
            }
        }
        
        # If user specified a strategy, respect it but optimize parameters
        if user_strategy == 'broad':
            strategy = strategies['broad'].copy()
        elif user_strategy == 'specific':
            strategy = strategies['focused'].copy()
        else:
            # Auto-select based on question features
            if features['scope'] == 'system_wide':
                strategy = strategies['broad'].copy()
            elif features['scope'] == 'specific_entity':
                strategy = strategies['focused'].copy()
            elif features['complexity'] == 'high':
                strategy = strategies['deep_dive'].copy()
            else:
                strategy = strategies['hybrid'].copy()
        
        # Fine-tune based on features
        if features['complexity'] == 'high':
            strategy['k'] = min(strategy['k'] + 2, 15)
        elif features['complexity'] == 'low':
            strategy['k'] = max(strategy['k'] - 1, 2)
        
        return strategy
    
    def _configure_retriever_with_strategy(self, retriever, strategy):
        """Configure retriever with optimal strategy parameters."""
        
        # Set search parameters
        retriever.search_kwargs = {"k": strategy['k']}
        
        # Could add more sophisticated configuration here:
        # - Similarity thresholds
        # - Representation type weighting
        # - Document filtering criteria
        
        return retriever
    
    def run(self, question, strategy='specific'):
        """Run the Multi-Representation agent with the given question and strategy."""
        print(f"--- RAG-based Code Expert Agent (Multi-Representation, Strategy: {strategy}) ---")
        
        # Step 1: Check if representations exist, build if needed
        print("--- Multi-Representation Setup ---")
        representations_path = self.storage_config["multi_representations"]
        if not os.path.exists(representations_path):
            print("Multi-representations not found. Building automatically...")
            print("This will take a while as it processes each file with the LLM...")
            success = self.build_representations()
            if not success:
                return
        
        # Step 2: Vectorization
        print("--- Vectorization ---")
        vectorizer = MultiRepresentationVectorizer(self.config)
        retriever = vectorizer.create_retriever()
        
        if retriever is None:
            return
        
        # Step 2: Configure retrieval based on adaptive strategy selection
        optimal_strategy = self._select_optimal_strategy(question, strategy)
        retriever = self._configure_retriever_with_strategy(retriever, optimal_strategy)
        
        print(f"Using {optimal_strategy['name']} strategy with k={optimal_strategy['k']}")
        
        # Step 3: Agent Orchestration
        print("--- Agent Orchestration ---")
        if self.provider == "wca":
            # For WCA, use the service directly
            wca_service = self._setup_provider()
            
            # Get relevant documents
            docs = retriever.invoke(question)
            print(f"Retrieved {len(docs)} documents using {strategy} strategy.")
            
            if not docs:
                print("No relevant documents found for the query.")
                return
            
            # Get analysis from WCA
            print("Sending query to WCA...")
            try:
                wca_response = wca_service.get_architectural_analysis(question, docs)
                answer_text = wca_service.extract_generated_text(wca_response) or "No response"

                qa_result = {
                    "question": question,
                    "answer": answer_text,
                    "source_documents": docs
                }
                
                self.response_generator.generate_response(qa_result)
                
            except Exception as e:
                print(f"Error calling WCA API: {e}")
        else:
            # For Ollama, use the chain
            qa_chain = self._create_qa_chain(retriever)
            self.response_generator.process_query(qa_chain, question)
        
        print("--- System Finished ---")


def main():
    """Main entry point for Multi-Representation prototype."""
    parser = argparse.ArgumentParser(description="RAG-based Code Expert Agent (Multi-Representation)")
    parser.add_argument("question", nargs="?", type=str, help="The question to ask the agent.")
    parser.add_argument("--strategy", choices=['broad', 'specific'], default='specific',
                       help="Retrieval strategy: 'broad' for overview questions, 'specific' for targeted questions")
    parser.add_argument("--build-representations", action="store_true", 
                       help="Build multi-level representations (slow, calls the LLM for each file)")
    parser.add_argument("--repo", help="Repository path to analyze")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Set repo path for data ingestion
    if args.repo:
        os.environ["REPO_PATH"] = args.repo
    
    # Create agent
    agent = MultiRepresentationAgent()
    
    if args.build_representations:
        # Build the representations
        print("This will take a while as it processes each file with the LLM...")
        success = agent.build_representations()
        if success:
            print("\nRepresentations built successfully! You can now run queries.")
            print("Example (broad): python multi_representation.py 'explain this repo' --strategy broad")
            print("Example (specific): python multi_representation.py 'how is user auth handled' --strategy specific")
        return
    
    if not args.question:
        print("Error: Please provide a question or use --build-representations to build representations first.")
        print("Usage: python multi_representation.py 'your question here' --strategy [broad|specific]")
        print("   or: python multi_representation.py --build-representations")
        return
    
    # Setup and pull required models
    setup_and_pull_models(config)
    
    # Run the agent
    agent.run(args.question, args.strategy)


if __name__ == "__main__":
    main()
