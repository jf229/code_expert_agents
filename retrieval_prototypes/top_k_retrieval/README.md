# Prototype: Top-K Retrieval

This prototype implements the baseline "Top-K Retrieval" strategy. It is the most common and straightforward approach for Retrieval-Augmented Generation (RAG) systems and serves as the foundation for the more advanced prototypes.

## Technical Deep Dive

This agent's intelligence is based on the `ParentDocumentRetriever` from the LangChain library.

1.  **Data Ingestion (`data_ingestion/main.py`):** The script scans the target repository and loads the full content of all source files. These are saved as "parent documents" in `raw_documents.pkl`.

2.  **Vectorization (`vectorization/main.py`):**
    *   The `ParentDocumentRetriever` is initialized.
    *   It uses a `RecursiveCharacterTextSplitter` to break down the raw parent documents into smaller, searchable "child" chunks.
    *   The text of these child chunks is converted into vector embeddings using the configured Ollama model (e.g., `bge-large-en-v1.5`) and stored in a ChromaDB vector store.
    *   A separate `InMemoryStore` is used to maintain a mapping between the child chunks and their parent documents. This is saved in `docstore.pkl`.

3.  **Orchestration (`agent_orchestration/main.py`):**
    *   When a question is asked, the retriever performs a similarity search against the child chunks in the vector store.
    *   It retrieves the `k` highest-ranking child chunks.
    *   Instead of returning these small chunks, it uses the `docstore` to retrieve the **full, original parent documents** associated with them.
    *   These complete source files are then passed to the final language model (either Ollama or WCA) with a detailed "Senior Software Architect" prompt to generate a structured, holistic answer.

## Use Case

This prototype is best for getting a **fast and accurate analysis of a specific, well-defined component** within the repository. Because it retrieves the full, unaltered source code of the most relevant files, it is excellent at answering questions where the detailed interaction between a few key files is important.

**Example Query:** `"What is the role of the PersistenceController and how does it interact with the DataService?"`

## How to Run

*All commands should be run from the root of the project.*

1.  **Navigate to Directory:**
    ```bash
    cd retrieval_prototypes/top_k_retrieval
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure:**
    *   Ensure `config.yaml` points to your target repository (`local_repo_path`).
    *   Set your desired `provider` (`ollama` or `wca`).
    *   Set the number of documents to retrieve with `top_k`.

4.  **Run the Full Pipeline:**
    The `main.py` script handles the entire data ingestion, vectorization, and questioning process in one go.
    ```bash
    python main.py "Your question here"
    ```