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

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from langchain_ollama import ChatOllama
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document

# Import shared services and central data ingestion
from shared import WCAService, ResponseGenerator, load_config, setup_and_pull_models, data_ingestion_main
from shared.data_ingestion import COMMON_LANGUAGE_EXTENSIONS



class GraphBuilder:
    """Builds a knowledge graph from repository source code."""
    
    def __init__(self, config):
        self.config = config
        self.storage_config = config["storage"]
    
    def build_graph(self, repo_path):
        """Build a knowledge graph from repository source code."""
        print("Detecting repository language...")
        
        # Detect the primary language of the repository
        language_counts = {}
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    language_counts['python'] = language_counts.get('python', 0) + 1
                elif file.endswith('.swift'):
                    language_counts['swift'] = language_counts.get('swift', 0) + 1
                elif file.endswith(('.js', '.ts')):
                    language_counts['javascript'] = language_counts.get('javascript', 0) + 1
        
        # Determine primary language
        if not language_counts:
            print("No source files found.")
            return nx.DiGraph()
        
        primary_language = max(language_counts, key=language_counts.get)
        print(f"Detected primary language: {primary_language} ({language_counts[primary_language]} files)")
        
        # Use appropriate parser
        if primary_language == 'swift' and TREE_SITTER_AVAILABLE:
            return self._build_swift_graph(repo_path)
        else:
            # For Python and other languages, use the basic file graph with entity extraction
            return self._build_basic_file_graph(repo_path)
    
    def _build_swift_graph(self, repo_path):
        """Build graph specifically for Swift repositories."""
        G = nx.DiGraph()
        
        try:
            SWIFT_LANGUAGE = Language(swift_language_capsule())
            parser = Parser()
            parser.language = SWIFT_LANGUAGE
            
            print("Building Swift knowledge graph...")
            swift_files = [
                os.path.join(root, file) 
                for root, _, files in os.walk(repo_path) 
                for file in files if file.endswith('.swift')
            ]
            
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
                                name = name_node.text.decode('utf-8')
                                node_id = f"{file_path}::{name}"
                                G.add_node(node_id, type='class', name=name, file=file_path)
                                G.add_edge(file_path, node_id, type='contains')
                        
                        elif node.type == 'function_declaration':
                            name_node = node.child_by_field_name('name')
                            if name_node:
                                name = name_node.text.decode('utf-8')
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
        
        # Use the same inclusion rules as shared ingestion: support both suffixes and filenames
        tokens = set(COMMON_LANGUAGE_EXTENSIONS.values())
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules', 'venv', '.venv'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                suffix = os.path.splitext(file)[1]
                # Include if matches known suffix or recognized filename
                if (suffix in tokens) or (file in tokens):
                    relative_path = os.path.relpath(file_path, repo_path)
                    
                    # Add file node with relative path for cleaner display
                    G.add_node(file_path, type='file', name=file, relative_path=relative_path)
                    
                    # For Python files, try to extract basic class/function info
                    if suffix == '.py':
                        self._add_python_entities(G, file_path, relative_path)
        
        print(f"Built basic graph with {G.number_of_nodes()} file nodes and {G.number_of_edges()} edges.")
        return G
    
    def _add_python_entities(self, graph, file_path, relative_path):
        """Add Python classes and functions to the graph using simple regex."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            import re
            
            # Find class definitions
            class_pattern = r'^class\s+(\w+)'
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                class_name = match.group(1)
                class_id = f"{file_path}::{class_name}"
                graph.add_node(class_id, type='class', name=class_name, file=file_path, relative_path=relative_path)
                graph.add_edge(file_path, class_id, type='contains')
            
            # Find function definitions (not inside classes for simplicity)
            func_pattern = r'^def\s+(\w+)'
            for match in re.finditer(func_pattern, content, re.MULTILINE):
                func_name = match.group(1)
                if not func_name.startswith('_'):  # Skip private functions
                    func_id = f"{file_path}::{func_name}"
                    graph.add_node(func_id, type='function', name=func_name, file=file_path, relative_path=relative_path)
                    graph.add_edge(file_path, func_id, type='contains')
        
        except Exception as e:
            # Silently skip files that can't be processed
            pass
    
    def save_graph(self, graph):
        """Save the graph to disk."""
        output_path = self.storage_config["code_graph"]
        with open(output_path, "wb") as f:
            pickle.dump(graph, f)
        
        print(f"Successfully built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        print(f"Graph saved to: {output_path}")


class GraphRetriever(BaseRetriever):
    """Custom retriever that fetches documents based on graph search."""
    
    graph: object = None  # Declare as a Pydantic field
    
    def __init__(self, graph, **kwargs):
        super().__init__(graph=graph, **kwargs)
    
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
        """Find relevant files based on query terms with semantic understanding."""
        relevant_files = set()
        query_terms = query.lower().split()
        
        # Define semantic mappings for common concepts
        concept_mappings = {
            'authentication': ['login', 'auth', 'signin', 'password', 'credential', 'token'],
            'authorization': ['permission', 'access', 'role', 'privilege', 'auth'],
            'data': ['model', 'schema', 'database', 'db', 'store', 'cache'],
            'api': ['endpoint', 'route', 'handler', 'controller', 'service'],
            'ui': ['view', 'component', 'render', 'display', 'interface'],
            'config': ['setting', 'configuration', 'option', 'preference'],
            'error': ['exception', 'catch', 'handle', 'fail', 'error']
        }
        
        # Expand query terms with semantic mappings
        expanded_terms = set(query_terms)
        for term in query_terms:
            for concept, related_terms in concept_mappings.items():
                if term in concept or concept in term:
                    expanded_terms.update(related_terms)
                elif term in related_terms:
                    expanded_terms.update(related_terms)
                    expanded_terms.add(concept)
        
        # Search through all nodes with expanded terms
        for node, data in self.graph.nodes(data=True):
            node_name = data.get('name', '').lower()
            node_id = str(node).lower()
            node_type = data.get('type', '')
            
            # Check if any expanded term matches node name or ID
            if any(term in node_name for term in expanded_terms) or any(term in node_id for term in expanded_terms):
                if data.get('type') == 'file':
                    relevant_files.add(node)
                elif 'file' in data:
                    relevant_files.add(data['file'])
        
        # If no matches found with expanded terms, try fuzzy matching
        if not relevant_files:
            for node, data in self.graph.nodes(data=True):
                node_name = data.get('name', '').lower()
                # Check for partial matches with original query terms
                if any(any(qt in node_name for qt in query_terms) for qt in query_terms if len(qt) > 3):
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
    
    def _load_graph(self):
        """Load the knowledge graph from disk."""
        graph_path = self.storage_config["code_graph"]
        if not os.path.exists(graph_path):
            print(f"Error: Graph file not found at {graph_path}")
            print("Please run with --build-graph first to create the knowledge graph.")
            return None
        
        with open(graph_path, "rb") as f:
            return pickle.load(f)
    
    def build_graph(self):
        """Build the knowledge graph."""
        print("--- Graph-Based Retrieval: Building Knowledge Graph ---")
        
        # Get repository path (environment variable takes precedence)
        repo_path = os.environ.get("REPO_PATH")
        if not repo_path:
            repo_path = self.config.get("repository", {}).get("local_path")
        
        if not repo_path or not os.path.isdir(repo_path):
            print("Error: Repository path not found. Set via --repo, REPO_PATH env var, or config.yaml")
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
        
        # Step 1: Check if graph exists, build if needed
        print("--- Knowledge Graph Setup ---")
        graph_path = self.storage_config["code_graph"]
        if not os.path.exists(graph_path):
            print("Knowledge graph not found. Building automatically...")
            success = self.build_graph()
            if not success:
                return
        
        # Load the graph
        graph = self._load_graph()
        if graph is None:
            return
        
        print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        
        # Step 2: Create graph retriever
        print("--- Creating Graph Retriever ---")
        retriever = GraphRetriever(graph)
        
        # Step 3: Agent Orchestration
        print("--- Agent Orchestration ---")
        
        # Get relevant documents from graph
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} documents from graph search.")
        
        if not docs:
            print("No relevant documents found for the query.")
            return
        
        # Use unified response generator (same as other agents)
        from shared import UnifiedResponseGenerator
        unified_generator = UnifiedResponseGenerator(self.config, prototype_name="Graph-Based Retrieval")
        unified_generator.generate_response_with_context(question, docs)
        
        print("--- System Finished ---")


def main():
    """Main entry point for Graph-Based Retrieval prototype."""
    parser = argparse.ArgumentParser(description="RAG-based Code Expert Agent (Graph-Based Retrieval)")
    parser.add_argument("question", nargs="?", type=str, help="The question to ask the agent.")
    parser.add_argument("--build-graph", action="store_true", help="Build the knowledge graph")
    parser.add_argument("--repo", help="Repository path to analyze")
    args = parser.parse_args()
    
    # Load configuration and pull models if needed
    config = load_config()
    setup_and_pull_models(config)
    
    # Set repo path for data ingestion
    if args.repo:
        os.environ["REPO_PATH"] = args.repo
    
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
    
    # Run the agent
    agent.run(args.question)


if __name__ == "__main__":
    main()
