#!/usr/bin/env python3
"""
Graph-Based Retrieval RAG Agent

The most powerful prototype for specific, targeted questions about code entities.
Uses NetworkX to build a knowledge graph of the codebase and retrieves 
relevant documents based on graph relationships.

Usage:
    # Step 1: Build the knowledge graph (run once)
    python graph_based_retrieval.py --build-graph
    
    # Step 2: Run queries
    python graph_based_retrieval.py "explain the LoginViewModel class"

Use Case: Answering precise questions about a specific class or function.
"""

import argparse
import os
import pickle
import yaml
from dotenv import load_dotenv
import networkx as nx

# Tree-sitter imports for parsing (currently supports Swift, can be extended)
try:
    from tree_sitter import Language, Parser
    from tree_sitter_swift import language as swift_language_capsule
    TREE_SITTER_AVAILABLE = True
except ImportError:
    print("Warning: tree-sitter not available. Graph building will be limited.")
    TREE_SITTER_AVAILABLE = False

# LangChain imports
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document

# Import shared services and central data ingestion
from shared_services import WCAService, ResponseGenerator, pull_ollama_model, load_config
from data_ingestion import main as data_ingestion_main


class GraphBuilder:
    """Builds a knowledge graph from repository source code."""
    
    def __init__(self, config):
        self.config = config
        self.storage_config = config["storage"]
    
    def build_graph(self, repo_path):
        """Build a simple knowledge graph from repository source code."""
        G = nx.DiGraph()
        
        if not TREE_SITTER_AVAILABLE:
            print("Tree-sitter not available. Building basic file graph...")
            return self._build_basic_file_graph(repo_path)
        
        # Currently supports Swift - can be extended for other languages
        try:
            SWIFT_LANGUAGE = Language(swift_language_capsule())
            parser = Parser()
            parser.language = SWIFT_LANGUAGE
            
            print("Scanning for Swift files...")
            swift_files = [
                os.path.join(root, file) 
                for root, _, files in os.walk(repo_path) 
                for file in files if file.endswith('.swift')
            ]
            print(f"Found {len(swift_files)} Swift files.")
            
            for file_path in swift_files:
                G.add_node(file_path, type='file')
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    tree = parser.parse(bytes(code, "utf8"))
                    
                    # Extract classes and functions
                    for node in tree.root_node.children:
                        if node.type == 'class_declaration':
                            name_node = node.child_by_field_name('name')
                            if name_node:
                                name = name_node.text.decode()
                                node_id = f"{file_path}::{name}"
                                G.add_node(node_id, type='class', name=name, file=file_path)
                                G.add_edge(file_path, node_id, type='contains')
                        
                        elif node.type == 'function_declaration':
                            name_node = node.child_by_field_name('name')
                            if name_node:
                                name = name_node.text.decode()
                                node_id = f"{file_path}::{name}"
                                G.add_node(node_id, type='function', name=name, file=file_path)
                                G.add_edge(file_path, node_id, type='contains')
                
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
                    continue
            
            return G
            
        except Exception as e:
            print(f"Error with Swift parsing: {e}")
            return self._build_basic_file_graph(repo_path)
    
    def _build_basic_file_graph(self, repo_path):
        """Build a basic file-based graph when tree-sitter is not available."""
        G = nx.DiGraph()
        
        print("Building basic file graph...")
        
        # Common source file extensions
        source_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.swift', '.kt', '.rb', '.go', '.rs'}
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if any(file.endswith(ext) for ext in source_extensions):
                    file_path = os.path.join(root, file)
                    G.add_node(file_path, type='file', name=file)
        
        print(f"Built basic graph with {G.number_of_nodes()} file nodes.")
        return G
    
    def save_graph(self, graph):
        """Save the graph to disk."""
        output_path = self.storage_config["code_graph"]
        with open(output_path, "wb") as f:
            pickle.dump(graph, f)
        
        print(f"Successfully built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        print(f"Graph saved to: {output_path}")


class GraphRetriever(BaseRetriever):
    """Custom retriever that fetches documents based on graph search."""
    
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
    
    def _get_relevant_documents(self, query: str):
        """Get relevant documents based on graph search."""
        relevant_files = self._retrieve_from_graph(query)
        retrieved_docs = []
        
        for file_path in relevant_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(page_content=content, metadata={'source': file_path})
                retrieved_docs.append(doc)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        return retrieved_docs
    
    def _retrieve_from_graph(self, query):
        """Find relevant files based on query terms."""
        relevant_files = set()
        query_terms = query.lower().split()
        
        for node, data in self.graph.nodes(data=True):
            node_name = data.get('name', '').lower()
            node_id = str(node).lower()
            
            # Check if any query term matches node name or ID
            if any(term in node_name for term in query_terms) or any(term in node_id for term in query_terms):
                if data.get('type') == 'file':
                    relevant_files.add(node)
                elif 'file' in data:
                    relevant_files.add(data['file'])
        
        return list(relevant_files)


class GraphBasedAgent:
    """Main Graph-Based Retrieval Agent."""
    
    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self.llm_config = self.config["llm"]
        self.storage_config = self.config["storage"]
        self.provider = self.llm_config.get("provider")
        self.response_generator = ResponseGenerator(prototype_name="Graph-Based Retrieval")
    
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
    
    def _load_graph(self):
        """Load the knowledge graph from disk."""
        graph_path = self.storage_config["code_graph"]
        if not os.path.exists(graph_path):
            print(f"Error: Graph file not found at {graph_path}")
            print("Please run with --build-graph first to create the knowledge graph.")
            return None
        
        with open(graph_path, "rb") as f:
            return pickle.load(f)
    
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
    
    def build_graph(self):
        """Build the knowledge graph."""
        print("--- Graph-Based Retrieval: Building Knowledge Graph ---")
        
        # Get repository path
        repo_path = self.config.get("repository", {}).get("local_path")
        if not repo_path or not os.path.isdir(repo_path):
            print("Error: 'repository.local_path' not found or invalid in config.yaml")
            return False
        
        print(f"Using repository at: {repo_path}")
        
        # Build graph
        graph_builder = GraphBuilder(self.config)
        graph = graph_builder.build_graph(repo_path)
        graph_builder.save_graph(graph)
        
        return True
    
    def run(self, question):
        """Run the Graph-Based retrieval agent with the given question."""
        print("--- RAG-based Code Expert Agent (Graph-Based Retrieval) ---")
        
        # Step 1: Load the graph
        print("--- Loading Knowledge Graph ---")
        graph = self._load_graph()
        if graph is None:
            return
        
        print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        
        # Step 2: Create graph retriever
        print("--- Creating Graph Retriever ---")
        retriever = GraphRetriever(graph)
        
        # Step 3: Agent Orchestration
        print("--- Agent Orchestration ---")
        if self.provider == "wca":
            # For WCA, use the service directly
            wca_service = self._setup_provider()
            
            # Get relevant documents from graph
            docs = retriever.get_relevant_documents(question)
            print(f"Retrieved {len(docs)} documents from graph search.")
            
            if not docs:
                print("No relevant documents found for the query.")
                return
            
            # Get analysis from WCA
            print("Sending query to WCA...")
            try:
                wca_response = wca_service.get_architectural_analysis(question, docs)
                
                # Format response for consistency
                qa_result = {
                    "question": question,
                    "answer": wca_response.get("choices", [{}])[0].get("message", {}).get("content", "No response"),
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
    """Main entry point for Graph-Based Retrieval prototype."""
    parser = argparse.ArgumentParser(description="RAG-based Code Expert Agent (Graph-Based Retrieval)")
    parser.add_argument("question", nargs="?", type=str, help="The question to ask the agent.")
    parser.add_argument("--build-graph", action="store_true", help="Build the knowledge graph")
    args = parser.parse_args()
    
    # Load configuration and pull models if needed
    config = load_config()
    
    # Create agent
    agent = GraphBasedAgent()
    
    if args.build_graph:
        # Build the knowledge graph
        success = agent.build_graph()
        if success:
            print("\nGraph built successfully! You can now run queries.")
            print("Example: python graph_based_retrieval.py 'explain the LoginViewModel class'")
        return
    
    if not args.question:
        print("Error: Please provide a question or use --build-graph to build the knowledge graph first.")
        print("Usage: python graph_based_retrieval.py 'your question here'")
        print("   or: python graph_based_retrieval.py --build-graph")
        return
    
    # Pull models if needed
    llm_model = config["llm"]["ollama_model"]
    embedding_model = config["embeddings"]["model"]
    
    pull_ollama_model(llm_model)
    pull_ollama_model(embedding_model)
    
    # Run the agent
    agent.run(args.question)


if __name__ == "__main__":
    main()