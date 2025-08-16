# Prototype: Graph--Based Retrieval

This prototype implements a "Graph-Based Retrieval" strategy. It is a highly advanced approach that understands the structural relationships within the code, leading to more intelligent and precise context retrieval for specific questions.

## Technical Deep Dive

This agent replaces semantic vector search with a structural search over a knowledge graph of the codebase.

1.  **Graph Creation (`graph_builder.py`):**
    *   This script uses the `tree-sitter` library and a set of language-specific grammar packages (`tree-sitter-swift`, etc.) to perform static analysis on the source code.
    *   It parses every source file to identify key code entities (classes, functions, etc.).
    *   It then uses the `networkx` library to build a directed graph where nodes are the files and the code entities they contain. Edges represent the "contains" relationship.
    *   The final graph is saved to `code_graph.gpickle`.
    *   *(Note: A more advanced implementation would also parse and create edges for relationships like `imports` and `calls` between entities.)*

2.  **Orchestration (`agent_orchestration/main.py`):**
    *   The pre-built `code_graph.gpickle` is loaded into memory.
    *   When a user asks a question, the `retrieve_from_graph` function performs a search over the nodes of the graph. It looks for nodes whose names or IDs contain the keywords from the user's query.
    *   It retrieves the file paths associated with all the matching nodes.
    *   The full content of these structurally-relevant files is then passed to the final language model (Ollama or WCA) with the "Senior Software Architect" prompt to generate a detailed, contextual answer.

## Use Case

This prototype is the most powerful and precise tool for answering **specific, targeted questions about a particular piece of code**. Because it understands the code's structure, it can reliably find the exact file where a class or function is defined, even if the user's query is slightly ambiguous.

**Example Query:** `"Explain the implementation of the LoginViewModel class."`

## How to Run

*All commands should be run from the root of the project.*

1.  **Navigate to Directory:**
    ```bash
    cd retrieval_prototypes/graph_based_retrieval
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure:**
    *   Ensure the `config.yaml` in this directory points to your target repository (`local_repo_path`).
    *   Set your desired `provider` (`ollama` or `wca`).

4.  **Step 1: Build the Graph**
    You must run the graph builder first to create the knowledge graph.
    ```bash
    python graph_builder.py
    ```

5.  **Step 2: Run the Agent**
    Now you can ask a specific question about a code entity.
    ```bash
    python main.py "Your specific question here"
    ```