# RAG-Based Code Expert Agent Prototypes

This project contains four distinct prototypes for a RAG-based (Retrieval-Augmented Generation) code expert system. Each prototype implements a different advanced retrieval strategy to answer high-level questions about a code repository.

## Simplified Architecture ✨

**New in v2:** This project has been completely refactored for simplicity and maintainability:

- **Single-file prototypes** - Each prototype is now a single Python file
- **Shared common functionality** - Common code consolidated in `shared_services.py`
- **No complex directory structures** - Flat, easy-to-navigate layout
- **Same powerful features** - All original functionality preserved

```
├── config.yaml                    # Centralized configuration
├── requirements.txt               # All dependencies  
├── data_ingestion.py              # Centralized data loading
├── shared_services.py             # Common WCA service, response generation, utilities
├── top_k_retrieval.py             # Top-K prototype (single file)
├── iterate_and_synthesize.py      # Iterate & Synthesize prototype (single file)
├── graph_based_retrieval.py       # Graph-based prototype (single file)
├── multi_representation.py        # Multi-representation prototype (single file)
└── README.md                      # This file
```

## Quick Start

### 1. Initial Setup

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Configure Repository:** Edit `config.yaml` and set `repository.local_path` to your code repository path.

**Configure API Key (if using WCA):** Create a `.env` file:
```bash
WCA_API_KEY=YOUR_API_KEY_HERE
```

**Choose Provider:** In `config.yaml`, set `llm.provider` to `"ollama"` or `"wca"`.

### 2. Run Any Prototype

All prototypes are now single files you can run directly:

---

**Fast & Simple:** Just run directly!

```bash
python top_k_retrieval.py "Provide an architectural analysis of the StravaService"
```

---

## Prototype 2: Iterate and Synthesize

This "MapReduce" agent analyzes every single file, making it extremely thorough but very slow.

**Use Case:** Generating a complete, repository-wide architectural overview.

```bash
python iterate_and_synthesize.py "Provide a comprehensive architectural analysis of this entire project"
```

---

## Prototype 3: Graph-Based Retrieval

The most powerful prototype for specific, targeted questions about code entities.

**Use Case:** Answering precise questions about a specific class or function.

```bash
# Just run directly! Graph builds automatically if needed
python graph_based_retrieval.py "explain the LoginViewModel class"

# Or build graph manually first (optional)
python graph_based_retrieval.py --build-graph
```

---

## Prototype 4: Multi-Representation Indexing

A hybrid agent that uses different strategies for different types of questions.

**Use Case:** Flexible analysis. Use `--strategy broad` for overview, `--strategy specific` for targeted questions.

```bash
# Just run directly! Representations build automatically if needed (slow first time)
python multi_representation.py "explain this repo" --strategy broad
python multi_representation.py "how is user authentication handled" --strategy specific

# Or build representations manually first (optional)
python multi_representation.py --build-representations
```

---

## What Changed in v2? 

✅ **Massive Simplification:**
- Reduced from **25+ files** to **8 files**
- Eliminated complex nested directories
- No more confusing import paths

✅ **Same Powerful Features:**
- All 4 retrieval strategies preserved
- All configuration options maintained  
- WCA and Ollama support unchanged

✅ **Much Better Developer Experience:**
- Single-file prototypes are easy to understand and modify
- Clear, direct commands 
- Consolidated shared functionality

This refactoring maintained 100% of the original functionality while making the codebase dramatically easier to work with.