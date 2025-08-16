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

**Features Enhanced Dynamic Prompting** - Automatically optimizes responses based on question type!

```bash
# Architectural questions get architecture-focused analysis
python top_k_retrieval.py "What is the overall system architecture?"

# Implementation questions get step-by-step explanations  
python top_k_retrieval.py "How does the authentication process work?"

# Entity-specific questions get detailed component analysis
python top_k_retrieval.py "What does the UserService class do?"

# Design rationale questions get decision-focused responses
python top_k_retrieval.py "Why was this database design chosen?"
```

---

## Prototype 2: Iterate and Synthesize

This "MapReduce" agent analyzes every single file, making it extremely thorough but very slow.

**Use Case:** Generating a complete, repository-wide architectural overview.

```bash
# Perfect for comprehensive analysis questions
python iterate_and_synthesize.py "Provide a complete architectural overview of this project"

# Great for understanding entire system workflows  
python iterate_and_synthesize.py "How does data flow through the entire application?"

# Ideal for broad technology stack analysis
python iterate_and_synthesize.py "What technologies and patterns are used throughout this codebase?"
```

---

## Prototype 3: Graph-Based Retrieval

The most powerful prototype for specific, targeted questions about code entities.

**Use Case:** Answering precise questions about a specific class or function.

```bash
# Perfect for exploring specific entities and their relationships
python graph_based_retrieval.py "explain the LoginViewModel class and its dependencies"

# Great for understanding component interactions
python graph_based_retrieval.py "how is the PaymentProcessor used throughout the system?"

# Ideal for finding related functionality
python graph_based_retrieval.py "what classes interact with the DatabaseManager?"

# Optional: Build graph manually first
python graph_based_retrieval.py --build-graph
```

---

## Prototype 4: Multi-Representation Indexing

A hybrid agent with **intelligent strategy selection** that adapts to your question automatically.

**Use Case:** Flexible analysis that automatically optimizes retrieval strategy.

```bash
# Broad questions automatically use wide search strategy
python multi_representation.py "explain this entire repository structure"

# Specific questions automatically use focused search strategy  
python multi_representation.py "how does the calculateTax function work?"

# Complex questions automatically use deep-dive strategy
python multi_representation.py "explain the complete user authentication and authorization flow"

# Manual strategy override (optional)
python multi_representation.py "system overview" --strategy broad
python multi_representation.py "specific class details" --strategy specific

# Optional: Build representations manually first  
python multi_representation.py --build-representations
```

---

## 🧠 Intelligent Features (New!)

### Dynamic Question Classification
All prototypes now automatically analyze your question and optimize their response:

| Question Type | Example | Response Focus |
|---------------|---------|----------------|
| **Architectural** | *"What's the system architecture?"* | Design patterns, component relationships, trade-offs |
| **Implementation** | *"How does login work?"* | Step-by-step flows, algorithms, technical details |
| **Entity-Specific** | *"What is UserController?"* | Purpose, interface, dependencies, usage |
| **Design Rationale** | *"Why use microservices?"* | Decision context, alternatives, benefits/drawbacks |
| **Data Flow** | *"How does data move through the system?"* | Processing stages, transformations, persistence |

### Adaptive Strategy Selection (Multi-Representation)
The multi-representation agent intelligently selects optimal strategies:

- **System-wide questions** → Automatically uses `broad` strategy (k=12+)
- **Specific entity questions** → Automatically uses `focused` strategy (k=3)  
- **Complex analysis questions** → Automatically uses `deep_dive` strategy (k=6+)
- **Moderate complexity** → Automatically uses `hybrid` strategy (k=8)

### Example Question → Strategy Mapping

```bash
# These automatically get optimal treatment:
"Explain the entire application" → broad strategy, architectural prompting
"What does calculateTax() do?" → focused strategy, entity-specific prompting  
"How does the complex payment flow work?" → deep_dive strategy, implementation prompting
"Why was Redis chosen for caching?" → hybrid strategy, rationale prompting
```

---

## What Changed in v2? 

✅ **Massive Simplification:**
- Reduced from **25+ files** to **8 files**
- Eliminated complex nested directories
- No more confusing import paths

✅ **🧠 Enhanced Intelligence:**
- **Dynamic question classification** with 6 question types
- **Adaptive prompting** optimized for each question type  
- **Intelligent strategy selection** in multi-representation agent
- **Auto-building** for graph and multi-representation prototypes

✅ **Same Powerful Features:**
- All 4 retrieval strategies preserved
- All configuration options maintained  
- WCA and Ollama support unchanged
- **Better answer quality** through smarter prompting

✅ **Much Better Developer Experience:**
- Single-file prototypes are easy to understand and modify
- Clear, direct commands with intelligent examples
- Consolidated shared functionality
- **Just ask questions naturally** - the agents adapt automatically

✅ **Quality Improvements:**
- Question-aware response structuring
- Context-optimized retrieval parameters
- Fallback mechanisms for robustness
- Comprehensive analysis document (AGENT_IMPROVEMENTS.md)

This refactoring maintained 100% of the original functionality while making the codebase dramatically easier to work with **and significantly smarter in how it responds to different types of questions**.