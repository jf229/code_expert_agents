# graph_builder.py
import os
import yaml
import networkx as nx
from tree_sitter import Language, Parser
from tree_sitter_swift import language as swift_language_capsule
import pickle
import sys

def build_graph(repo_path):
    """
    Builds a simple knowledge graph from a repository's Swift source code.
    """
    SWIFT_LANGUAGE = Language(swift_language_capsule())
    parser = Parser()
    parser.language = SWIFT_LANGUAGE

    G = nx.DiGraph()

    print("Scanning for Swift files...")
    swift_files = [os.path.join(root, file) for root, _, files in os.walk(repo_path) for file in files if file.endswith('.swift')]
    print(f"Found {len(swift_files)} Swift files.")

    for file_path in swift_files:
        G.add_node(file_path, type='file')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = parser.parse(bytes(code, "utf8"))
        
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
    return G

def main():
    print("--- Simple Swift Graph Builder ---")
    
    config_path = "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Root configuration file not found at {config_path}")
        sys.exit(1)

    repo_path = config.get("repository", {}).get("local_path")
    if not repo_path or not os.path.isdir(repo_path):
        print(f"Error: 'repository.local_path' not found or invalid in {config_path}")
        sys.exit(1)

    print(f"Using repository at: {repo_path}")
    graph = build_graph(repo_path)
    
    output_path = config["storage"]["code_graph"]
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
        
    print(f"Successfully built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Graph saved to: {output_path}")

if __name__ == "__main__":
    main()