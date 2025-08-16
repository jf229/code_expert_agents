# CLAUDE.md - RAG Code Expert Agents Architecture Analysis

## Project Overview

This project implements four distinct RAG-based (Retrieval-Augmented Generation) code expert prototypes for analyzing and answering questions about code repositories. Each prototype uses different retrieval strategies while sharing common infrastructure.

## Critical Issues & Recommendations

### 🚨 Major Code Duplication Issues

The codebase suffers from extensive code duplication that should be addressed immediately:

1. **100% Identical Files**:
   - `iterate_and_synthesize/data_ingestion/main.py` and `graph_based_retrieval/data_ingestion/main.py` and `multi_representation_indexing/data_ingestion/main.py` are identical
   - `iterate_and_synthesize/vectorization/main.py` and `graph_based_retrieval/vectorization/main.py` are identical
   - All 4 `reasoning_response_generation/main.py` files are 99% identical
   - All 4 `wca_service.py` files contain mostly duplicate WCA API handling code

2. **Near-Identical Code Patterns** (60-80% duplication):
   - Agent orchestration modules share extensive configuration and provider setup code
   - Vectorization modules follow nearly identical patterns for Chroma/Ollama setup
   - Main.py files across prototypes use identical argument parsing and pipeline orchestration

## Architectural Improvements

### 1. Consolidate Shared Infrastructure

#### Current Structure Problems:
```
retrieval_prototypes/
├── top_k_retrieval/
│   ├── main.py
│   ├── wca_service.py          # Duplicate
│   ├── vectorization/main.py   # Similar patterns
│   ├── agent_orchestration/main.py  # 60% shared code
│   └── reasoning_response_generation/main.py  # 99% duplicate
├── iterate_and_synthesize/
│   ├── [same structure with mostly duplicate files]
├── graph_based_retrieval/
│   ├── [same structure with mostly duplicate files]
└── multi_representation_indexing/
    ├── [same structure with mostly duplicate files]
```

#### Recommended Improved Structure:
```
├── config.yaml                 # ✅ Already centralized
├── requirements.txt            # ✅ Already centralized
├── data_ingestion.py          # ✅ Already centralized (but not used by all)
├── shared/                     # 🆕 New shared infrastructure
│   ├── __init__.py
│   ├── base/
│   │   ├── config_manager.py   # Centralized config loading
│   │   ├── provider_factory.py # WCA/Ollama provider setup
│   │   └── base_orchestrator.py # Common orchestration patterns
│   ├── services/
│   │   ├── wca_service.py      # Single WCA service implementation
│   │   └── llm_service.py      # LLM abstraction layer
│   ├── vectorization/
│   │   ├── base_vectorizer.py  # Common vectorization patterns
│   │   └── vectorization_strategies.py # Different retrieval strategies
│   └── response_generation/
│       ├── response_generator.py # Unified response generation
│       └── formatters.py       # Different output formats
└── prototypes/                 # 🆕 Renamed and simplified
    ├── top_k_retrieval.py      # Single file per prototype
    ├── iterate_and_synthesize.py
    ├── graph_based_retrieval.py
    └── multi_representation_indexing.py
```

### 2. Create Unified Base Classes

#### A. Base Orchestrator Pattern
```python
# shared/base/base_orchestrator.py
class BaseAgentOrchestrator:
    def __init__(self, config_path="config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.provider = self._create_provider()
    
    def _create_provider(self):
        if self.config_manager.is_wca():
            return WCAService(self.config_manager.get_api_key())
        else:
            return OllamaService(self.config_manager.get_ollama_config())
    
    def create_qa_chain(self, retriever):
        # Common QA chain creation logic
        pass
```

#### B. Strategy Pattern for Retrievers
```python
# shared/vectorization/vectorization_strategies.py
class VectorizationStrategy:
    def create_retriever(self, documents, config):
        raise NotImplementedError

class TopKStrategy(VectorizationStrategy):
    def create_retriever(self, documents, config):
        # Top-K specific implementation
        pass

class GraphStrategy(VectorizationStrategy):
    def create_retriever(self, documents, config):
        # Graph-based specific implementation
        pass
```

### 3. Eliminate Architecture Redundancy

#### Current Issues:
- Each prototype has 4-5 separate modules doing similar work
- Extensive file system overhead (25 Python files vs potential 8-10)
- Configuration scattered across modules instead of centralized flow

#### Recommended Simplification:
Each prototype should be a single file that:
1. Imports shared infrastructure
2. Configures strategy-specific behavior
3. Executes the pipeline

Example simplified prototype:
```python
# prototypes/top_k_retrieval.py
from shared.base import BaseAgentOrchestrator
from shared.vectorization import TopKStrategy
from shared.response_generation import ResponseGenerator

class TopKRetrieval(BaseAgentOrchestrator):
    def __init__(self):
        super().__init__()
        self.strategy = TopKStrategy()
    
    def run(self, question):
        documents = self.load_documents()
        retriever = self.strategy.create_retriever(documents, self.config_manager.config)
        qa_chain = self.create_qa_chain(retriever)
        return ResponseGenerator().generate_response(qa_chain, question)

if __name__ == "__main__":
    # Argument parsing and execution
    pass
```

### 4. Configuration Standardization

#### Current Problems:
- Mixed configuration schemas across prototypes
- Some use centralized config, others use hardcoded paths
- Inconsistent environment variable handling

#### Recommended Solution:
```yaml
# config.yaml - Enhanced structure
repository:
  local_path: "/path/to/repo"
  remote_url: "https://github.com/user/repo"  # fallback

providers:
  llm:
    provider: "wca"  # or "ollama"
    ollama_model: "granite3.2:8b"
    wca_api_key_env: "WCA_API_KEY"
  
  embeddings:
    model: "nomic-embed-text"
    provider: "ollama"  # always local for privacy

storage:
  vector_store: "vector_store"
  documents: "raw_documents.pkl"
  docstore: "docstore.pkl"
  graph: "code_graph.gpickle"
  multi_representations: "multi_representations.pkl"

strategies:
  top_k:
    k: 5
    chunk_size: 1000
  
  graph:
    max_depth: 3
    similarity_threshold: 0.7
  
  multi_representation:
    broad_strategy_k: 10
    specific_strategy_k: 5
```

## Implementation Priority

### Phase 1: Critical Duplicates (Immediate - 1-2 days)
1. **Delete identical files**: Remove duplicate data_ingestion and vectorization files
2. **Consolidate WCA service**: Create single shared WCA service
3. **Unify response generation**: Single response generation module
4. **Update imports**: Modify prototypes to use shared modules

**Impact**: Reduces codebase by ~60%, eliminates maintenance overhead

### Phase 2: Infrastructure Refactoring (1 week)
1. **Create shared base classes**: BaseOrchestrator, ConfigManager, ProviderFactory
2. **Implement strategy pattern**: VectorizationStrategy, ResponseStrategy
3. **Simplify prototype files**: Convert to single-file implementations
4. **Standardize configuration**: Unified config schema

**Impact**: Improves maintainability, enables easier testing, cleaner architecture

### Phase 3: Advanced Features (2 weeks)
1. **Enhanced error handling**: Structured exception handling, retry logic
2. **Logging infrastructure**: Replace print statements with proper logging
3. **Testing framework**: Unit and integration tests for shared modules
4. **Performance optimization**: Caching, async operations where beneficial

**Impact**: Production-ready codebase, easier debugging, better observability

## Benefits of Refactoring

### Code Quality Improvements
- **75% reduction in duplicate code** (from ~25 files to ~10 files)
- **Single source of truth** for common functionality
- **Consistent error handling** across all prototypes
- **Easier testing** with isolated, mockable components

### Development Experience
- **Faster feature development** - add to one place, benefits all prototypes
- **Easier debugging** - centralized logging and error handling
- **Better documentation** - focused on unique features rather than duplicated code
- **Simplified onboarding** - clear separation of shared vs. prototype-specific code

### Maintenance Benefits
- **Bug fixes propagate automatically** to all prototypes
- **Configuration changes** apply consistently
- **Dependency updates** managed centrally
- **Performance improvements** benefit entire system

## Testing Strategy

### Unit Tests Structure
```
tests/
├── shared/
│   ├── test_config_manager.py
│   ├── test_wca_service.py
│   ├── test_vectorization_strategies.py
│   └── test_response_generation.py
├── integration/
│   ├── test_end_to_end_flows.py
│   └── test_provider_switching.py
└── prototypes/
    ├── test_top_k_retrieval.py
    ├── test_iterate_synthesize.py
    ├── test_graph_retrieval.py
    └── test_multi_representation.py
```

### Test Commands
```bash
# Run all tests
python -m pytest tests/

# Run shared module tests
python -m pytest tests/shared/

# Run specific prototype tests
python -m pytest tests/prototypes/test_top_k_retrieval.py

# Run with coverage
python -m pytest tests/ --cov=shared --cov=prototypes
```

## Development Commands

### Lint and Type Check
```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black shared/ prototypes/

# Lint code
flake8 shared/ prototypes/

# Type check
mypy shared/ prototypes/
```

### Running Prototypes (After Refactoring)
```bash
# Top-K Retrieval
python prototypes/top_k_retrieval.py "Analyze the authentication system"

# Iterate and Synthesize
python prototypes/iterate_and_synthesize.py "Provide comprehensive project overview"

# Graph-Based Retrieval
python prototypes/graph_based_retrieval.py "Explain the LoginViewModel class"

# Multi-Representation Indexing
python prototypes/multi_representation_indexing.py "How is user data managed?" --strategy specific
```

## Long-term Vision

This refactoring sets the foundation for:

1. **Plugin Architecture**: Easy addition of new retrieval strategies
2. **API Interface**: RESTful API wrapping the prototype functionality  
3. **UI Integration**: Web interface for non-technical users
4. **Performance Monitoring**: Metrics and observability for production use
5. **Model Flexibility**: Support for additional LLM providers (OpenAI, Anthropic, etc.)

The goal is to transform this from a research prototype codebase into a maintainable, extensible platform for code analysis and understanding.