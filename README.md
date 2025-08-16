# RAG-Based Code Expert Agent Prototypes

This project contains four distinct prototypes for a modular, RAG-based (Retrieval-Augmented Generation) code expert system. Each prototype implements a different advanced retrieval strategy to answer high-level questions about a code repository.

## Centralized Architecture

This project uses a single, centralized `config.yaml` file and a single `requirements.txt` file located in the root of the project. This makes it easy to manage the settings and dependencies for all four prototypes from one place. All data artifacts (vector stores, graphs, etc.) are also stored in the root directory.

## How to Use the Prototypes

### 1. Initial Setup

*   **Install Dependencies:** From the root of the project, run:
    ```bash
    pip install -r requirements.txt
    ```
*   **Configure Repository:** Open the root `config.yaml` and set the `repository.local_path` to the absolute path of the code repository you want to analyze.
*   **Configure API Key:** Create a `.env` file in the root of the project by copying the `.env.example` template. Add your Watson Code Assistant API key to this file.
    ```
    WCA_API_KEY=YOUR_API_KEY_HERE
    ```
*   **Choose a Provider:** In `config.yaml`, set the `llm.provider` to either `"ollama"` or `"wca"`. If using Ollama, ensure your local Ollama server is running.

### 2. Running a Prototype

All commands should be run from the **root of the project**.

---

### Prototype 1: Top-K Retrieval

This is the baseline approach. It's fast and effective for clear, specific questions. It performs data ingestion and vectorization automatically.

**Use Case:** Getting a quick, high-level analysis based on the most relevant files.

**Command:**
```bash
python retrieval_prototypes/top_k_retrieval/main.py "Provide an architectural analysis of the StravaService"
```

---

### Prototype 2: Iterate and Synthesize

This "MapReduce" agent analyzes every single file, making it extremely thorough but very slow.

**Use Case:** Generating a complete, repository-wide architectural overview.

**Command:**
```bash
python retrieval_prototypes/iterate_and_synthesize/main.py "Provide a comprehensive architectural analysis of this entire project"
```

---

### Prototype 3: Graph-Based Retrieval

The most powerful prototype for specific, targeted questions about code entities.

**Use Case:** Answering precise questions about a specific class or function.

**Commands:**
```bash
# Step 1: Build the knowledge graph
python retrieval_prototypes/graph_based_retrieval/graph_builder.py

# Step 2: Run the agent with a specific question
python retrieval_prototypes/graph_based_retrieval/main.py "explain the LoginViewModel class"
```

---

### Prototype 4: Multi-Representation Indexing

A hybrid agent that uses different strategies for different types of questions.

**Use Case:** Flexible analysis. Use `--strategy broad` for a high-level overview, and `--strategy specific` for targeted questions.

**Commands:**
```bash
# Step 1: Build the multi-level representations (slow, calls the LLM for each file)
python retrieval_prototypes/multi_representation_indexing/representation_builder.py

# Step 2: Build the vector store
python retrieval_prototypes/multi_representation_indexing/vectorization/main.py

# Step 3: Run the Agent
# For a broad overview:
python retrieval_prototypes/multi_representation_indexing/main.py "explain this repo" --strategy broad
# For a specific question:
python retrieval_prototypes/multi_representation_indexing/main.py "how is user authentication handled" --strategy specific
```