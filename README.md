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

**Configure API Keys:** Create a `.env` file with your chosen provider's API key:
```bash
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# For Claude (Anthropic)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Gemini (Google)
GOOGLE_API_KEY=your_google_api_key_here

# For WCA (Watson Code Assistant)
WCA_API_KEY=your_wca_api_key_here
```

**Choose Provider:** In `config.yaml`, set `llm.provider` to one of:
- `"ollama"` (local, free)
- `"openai"` (GPT-4, GPT-3.5)
- `"claude"` (Claude 3 Opus, Sonnet, Haiku)
- `"gemini"` (Gemini Pro)
- `"wca"` (Watson Code Assistant)

**Repository Configuration:** You can specify repositories in three ways:
1. **Command line (recommended)**: `--repo /path/to/your/repo`
2. **Environment variable**: `REPO_PATH=/path/to/your/repo`
3. **Config file**: Edit `config.yaml` and set `repository.local_path`

### 2. Run Any Prototype

All prototypes are now single files you can run directly:

---

**Features Enhanced Dynamic Prompting** - Automatically optimizes responses based on question type!

```bash
# Architectural questions get architecture-focused analysis
python top_k_retrieval.py "What is the overall system architecture?" --repo /path/to/repo

# Implementation questions get step-by-step explanations  
python top_k_retrieval.py "How does the authentication process work?" --repo /path/to/repo

# Entity-specific questions get detailed component analysis
python top_k_retrieval.py "What does the UserService class do?" --repo /path/to/repo

# Design rationale questions get decision-focused responses
python top_k_retrieval.py "Why was this database design chosen?" --repo /path/to/repo

# Or use environment variable for multiple queries
export REPO_PATH=/path/to/your/repo
python top_k_retrieval.py "What is the system architecture?"
```

---

## Prototype 2: Iterate and Synthesize

This "MapReduce" agent analyzes every single file, making it extremely thorough but very slow.

**Use Case:** Generating a complete, repository-wide architectural overview.

```bash
# Perfect for comprehensive analysis questions
python iterate_and_synthesize.py "Provide a complete architectural overview of this project" --repo /path/to/repo

# Great for understanding entire system workflows  
python iterate_and_synthesize.py "How does data flow through the entire application?" --repo /path/to/repo

# Ideal for broad technology stack analysis
python iterate_and_synthesize.py "What technologies and patterns are used throughout this codebase?" --repo /path/to/repo
```

---

## Prototype 3: Graph-Based Retrieval

The most powerful prototype for specific, targeted questions about code entities.

**Use Case:** Answering precise questions about a specific class or function.

```bash
# Perfect for exploring specific entities and their relationships
python graph_based_retrieval.py "explain the LoginViewModel class and its dependencies" --repo /path/to/repo

# Great for understanding component interactions
python graph_based_retrieval.py "how is the PaymentProcessor used throughout the system?" --repo /path/to/repo

# Ideal for finding related functionality
python graph_based_retrieval.py "what classes interact with the DatabaseManager?" --repo /path/to/repo

# Optional: Build graph manually first
python graph_based_retrieval.py --build-graph --repo /path/to/repo
```

---

## Prototype 4: Multi-Representation Indexing

A hybrid agent with **intelligent strategy selection** that adapts to your question automatically.

**Use Case:** Flexible analysis that automatically optimizes retrieval strategy.

```bash
# Broad questions automatically use wide search strategy
python multi_representation.py "explain this entire repository structure" --repo /path/to/repo

# Specific questions automatically use focused search strategy  
python multi_representation.py "how does the calculateTax function work?" --repo /path/to/repo

# Complex questions automatically use deep-dive strategy
python multi_representation.py "explain the complete user authentication and authorization flow" --repo /path/to/repo

# Manual strategy override (optional)
python multi_representation.py "system overview" --strategy broad --repo /path/to/repo
python multi_representation.py "specific class details" --strategy specific --repo /path/to/repo

# Optional: Build representations manually first  
python multi_representation.py --build-representations --repo /path/to/repo
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
python multi_representation.py "Explain the entire application" --repo /path/to/repo
# → broad strategy, architectural prompting

python multi_representation.py "What does calculateTax() do?" --repo /path/to/repo  
# → focused strategy, entity-specific prompting

python multi_representation.py "How does the complex payment flow work?" --repo /path/to/repo
# → deep_dive strategy, implementation prompting

python multi_representation.py "Why was Redis chosen for caching?" --repo /path/to/repo
# → hybrid strategy, rationale prompting
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