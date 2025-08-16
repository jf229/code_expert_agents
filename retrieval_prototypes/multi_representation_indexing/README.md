# Prototype: Multi-Representation Indexing

This prototype implements a "Multi-Representation Indexing" strategy. It is a sophisticated hybrid agent that improves retrieval accuracy by searching over multiple levels of abstraction. It intelligently chooses between a broad or specific analysis strategy based on a command-line argument.

## Technical Deep Dive

This agent's core innovation is to index not just the raw code, but also AI-generated summaries and hypothetical questions related to it.

1.  **Representation Generation (`representation_builder.py`):**
    *   This script is the first step in the pipeline. It iterates through every source file in the repository.
    *   For each file, it makes two calls to the configured language model (Ollama or WCA):
        1.  To generate a concise summary of the file's purpose.
        2.  To generate a list of 3-5 hypothetical questions the file could answer.
    *   The original content, the summary, and the questions for every file are saved into a single `multi_representations.pkl` file.

2.  **Vectorization (`vectorization/main.py`):**
    *   This script uses LangChain's `MultiVectorRetriever`.
    *   It populates a ChromaDB vector store with the embeddings of the **summaries and hypothetical questions**.
    *   It populates a separate `InMemoryStore` (`docstore.pkl`) with the **full, original content** of the source files.
    *   Crucially, it creates a mapping so that each summary and question in the vector store points to the original document in the docstore.

3.  **Orchestration (`agent_orchestration/main.py`):**
    *   The agent accepts a `--strategy` flag (`broad` or `specific`).
    *   **If `--strategy broad`:** It bypasses the retriever entirely. It loads the `multi_representations.pkl` file, concatenates all the summaries, and sends them to the LLM with the "Senior Software Architect" prompt for a full repository analysis (the "Summarize the Summaries" method).
    *   **If `--strategy specific`:** It uses the `MultiVectorRetriever`. When a user asks a question, the retriever searches for the most similar **summaries and questions** in the vector store. It then uses the mapping to retrieve the **full, original content** of the associated documents and sends them to the LLM for a detailed answer.

## Use Case

This prototype is a flexible, powerful tool for a variety of queries.

*   **Broad Analysis:** Use the `--strategy broad` for a complete, repository-wide overview, similar to the "Iterate and Synthesize" prototype but with a more efficient final step.
    **Example:** `python main.py "explain this repo" --strategy broad`
*   **Conceptual Questions:** Use the `--strategy specific` for questions that are more conceptual or use different terminology than the code. The search over summaries and questions is very effective at finding the right files in these cases.
    **Example:** `python main.py "how is user authentication handled" --strategy specific`

## How to Run

*All commands should be run from the root of the project.*

1.  **Navigate to Directory:**
    ```bash
    cd retrieval_prototypes/multi_representation_indexing
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure:**
    *   Ensure `config.yaml` points to your target repository (`local_repo_path`).
    *   Set your desired `provider` (`ollama` or `wca`).

4.  **Step 1: Ingest Documents**
    ```bash
    python data_ingestion/main.py
    ```

5.  **Step 2: Build Representations**
    This step is slow as it calls the LLM for every file.
    ```bash
    python representation_builder.py
    ```

6.  **Step 3: Build the Vector Store**
    ```bash
    python vectorization/main.py
    ```

7.  **Step 4: Run the Agent**
    Choose your strategy with the `--strategy` flag.
    ```bash
    # For a broad overview
    python main.py "explain this repo" --strategy broad

    # For a specific question
    python main.py "how does authentication work" --strategy specific
    ```