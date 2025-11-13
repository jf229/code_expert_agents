# Planned Improvements & Feature Ideas

## 1. Model Fine-Tuning from Stored Analysis (Priority: High)

### Overview
Currently, the system uses RAG (Retrieval-Augmented Generation) with custom prompting - we store comprehensive analysis documents and inject them into prompts when answering questions. This feature proposes fine-tuning a base LLM model on the stored analysis data to create a specialized model for each project.

### Current Approach (RAG)
- **IterateAndSynthesizeAgent**: Analyzes entire codebase once, stores comprehensive analysis
- **QueryAgent**: Loads stored analysis and includes it in prompt context
- **Pros**: Flexible, easy to update analysis by re-running agent
- **Cons**: Large context windows needed, slower inference with large documents, less internalized knowledge

### Proposed Approach (Fine-Tuning)
Fine-tune an Ollama-compatible base model (e.g., granite3.2:8b) using the stored comprehensive analysis data from each project.

#### Implementation Design

**Phase 1: Training Data Preparation**
- Use the stored `comprehensive_analysis.pkl` from IterateAndSynthesizeAgent
- Convert analysis into fine-tuning format (instruction-response pairs)
- Example format:
  ```
  Instruction: What are the main components of [Project]?
  Response: [Extracted from comprehensive_analysis]

  Instruction: What design patterns are used?
  Response: [Extracted from comprehensive_analysis]
  ```
- Generate synthetic QA pairs from the analysis using the base model

**Phase 2: Model Fine-Tuning Pipeline**
- Create a fine-tuning module that:
  - Takes the project's stored analysis as input
  - Generates training data with synthetic QA pairs
  - Fine-tunes the base model using tools like:
    - `ollama serve` with custom training (if supported)
    - Or use external fine-tuning tools (llama.cpp, LLaMA Factory, etc.)
  - Saves the fine-tuned model as project-specific (e.g., `granite3.2-ansible-fixer:8b`)

**Phase 3: Model Registry & Selection**
- Maintain a registry of available models per project:
  ```
  project_storage/
  ├── ansible-lint-fixer/
  │   ├── comprehensive_analysis.pkl
  │   ├── model_registry.json  # {"fine_tuned_model": "granite3.2-ansible-fixer:8b"}
  │   └── training_data/
  ├── rh_web/
  │   ├── comprehensive_analysis.pkl
  │   ├── model_registry.json
  │   └── training_data/
  ```

**Phase 4: QueryAgent Enhancement**
- Detect if a fine-tuned model exists for the project
- Use fine-tuned model if available, fallback to base model + RAG
- Commands:
  ```bash
  python agents/query_agent.py /path/to/repo "What is this?"          # Uses fine-tuned if available
  python agents/query_agent.py /path/to/repo "What is this?" --rag    # Force RAG mode
  python agents/query_agent.py /path/to/repo "What is this?" --retrain # Rebuild fine-tuned model
  ```

#### Benefits
1. **Internalized Knowledge**: Model has learned project architecture, not just retrieving it
2. **Faster Inference**: No need to include large documents in context window
3. **Better Handling**: Model can reason about complex architectural questions
4. **Project-Specific**: Each project gets specialized knowledge
5. **Offline Capable**: Fine-tuned model works without fetching documents

#### Trade-offs
1. **Training Time**: Fine-tuning takes computational resources
2. **Storage**: Store multiple project-specific models
3. **Updates**: Need to re-train when codebase changes significantly
4. **Complexity**: More moving parts than simple RAG

#### Recommendation
- **Start with**: RAG mode (current implementation) as the default
- **Future**: Add fine-tuning as opt-in feature with `--train-model` flag
- **Gradual Rollout**: Fine-tune for frequently-accessed projects first

---

## 2. Interactive Chat Sessions

### Overview
Create an interactive chat mode where users can have multi-turn conversations about their codebase.

### Current State
- One-shot questions with QueryAgent
- Single response per query

### Enhancement
```bash
python agents/chat.py /path/to/repo
> What is this project about?
[Response]
> Tell me more about the architecture
[Response using conversation history]
> How does authentication work?
[Response with context from previous questions]
```

### Implementation
- Maintain conversation history
- Use fine-tuned model or RAG with conversation context
- Support commands like `/analyze`, `/save-session`, `/export-conversation`

---

## 3. Multi-Project Analysis

### Overview
Analyze relationships and dependencies between multiple projects.

### Features
- Compare architectures across projects
- Identify shared patterns
- Suggest refactoring opportunities

---

## 4. Document Generation

### Overview
Generate project documentation automatically from the comprehensive analysis.

### Output Formats
- Markdown README
- Architecture diagrams (PlantUML/Mermaid)
- API documentation
- Component relationship charts

---

## 5. Continuous Analysis

### Overview
Instead of one-time analysis, continuously update analysis as codebase changes.

### Implementation
- Watch for file changes
- Incrementally update comprehensive analysis
- Keep fine-tuned models fresh
- Integration with Git webhooks

---

## 6. Web Interface

### Overview
Build a web UI for easy access to code analysis without CLI.

### Features
- Browse projects
- Ask questions
- View generated documentation
- Monitor analysis updates
- Export reports

---

## 7. Performance Optimization

### Current Bottlenecks
- IterateAndSynthesizeAgent: ~8-10 seconds per file on 21-file project = 3+ minutes
- Larger projects (59+ files) take even longer

### Optimization Ideas
1. **Parallel Processing**: Process multiple files simultaneously
2. **Incremental Analysis**: Only analyze changed files
3. **Caching**: Cache file summaries and reuse across queries
4. **Smarter Chunking**: Better code segmentation for faster analysis
5. **Compression**: Compress analysis for faster loading

---

## 8. Advanced Retrieval Strategies

### Current Agents
- Top-K Retrieval
- Graph-Based Retrieval
- Multi-Representation Retrieval
- Comprehensive Analysis

### Enhancements
- Hybrid retrieval combining multiple strategies
- Semantic similarity improvements
- Cross-project code pattern search
- Anomaly detection in codebase structure

---

## Priority Roadmap

1. **Now**: Finish testing all agents (rh_web, bikecheck)
2. **Next**: Fine-tuning infrastructure (Phase 1-2)
3. **Then**: Interactive chat with conversation history
4. **Later**: Web interface, continuous analysis, documentation generation
