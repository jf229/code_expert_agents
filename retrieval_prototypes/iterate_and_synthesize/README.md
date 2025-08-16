# Prototype: Iterate and Synthesize (MapReduce)

This prototype implements the "Iterate and Synthesize" strategy, also known as a MapReduce or Hierarchical Summarization approach. It is designed for maximum completeness, ensuring that information from every single file is considered.

## Technical Deep Dive

This agent does not use a vector store for retrieval. Instead, it follows a two-step "MapReduce" process to analyze the entire repository.

1.  **Data Ingestion (`data_ingestion/main.py`):** The script scans the target repository and loads the full content of all source files, saving them to `raw_documents.pkl`.

2.  **Orchestration (`agent_orchestration/main.py`):**
    *   **Map Step:** The orchestrator loads the raw documents. It then iterates through **every single document**, making an individual call to the configured language model (Ollama or WCA) for each one with a prompt to "Summarize the purpose and key components of this file." A progress bar is displayed during this phase.
    *   **Reduce Step:** All of the individual file summaries are concatenated into a single, large text block. A final call is made to the language model with the powerful "Senior Software Architect" prompt, asking it to perform a holistic analysis based on the provided summaries.

3.  **Generation:** The final, synthesized answer is returned to the user.

## Use Case

This prototype is best for generating a **complete, high-level architectural overview of an entire repository**, especially when you need to be certain that no file has been overlooked. It is less effective for specific, detailed questions, as the final analysis is based on summaries, not the original code.

**Example Query:** `"Provide a comprehensive architectural analysis of this entire project."`

## How to Run

*All commands should be run from the root of the project.*

1.  **Navigate to Directory:**
    ```bash
    cd retrieval_prototypes/iterate_and_synthesize
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure:**
    *   Ensure `config.yaml` points to your target repository (`local_repo_path`).
    *   Set your desired `provider` (`ollama` or `wca`).

4.  **Run the Full Pipeline:**
    The `main.py` script handles the data ingestion and the full MapReduce process. **Be aware that this will be very slow**, as it makes an API call for every file.
    ```bash
    python main.py "Your high-level question here"
    ```