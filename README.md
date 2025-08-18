# RAG Code Expert Agents

A collection of AI-powered code analysis agents that use different Retrieval-Augmented Generation (RAG) strategies to answer questions about code repositories. Each agent specializes in different types of analysis - from quick lookups to comprehensive architectural reviews.

## What This Does

This system analyzes code repositories using AI to answer natural language questions like:
- "How does authentication work in this codebase?"
- "What is the overall system architecture?"  
- "Explain the UserController class and its dependencies"
- "How does data flow through the payment system?"

It includes 4 core RAG agents, each using different retrieval strategies, plus advanced orchestration and evaluation tools.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Pull local models (recommended)
ollama pull granite3.2:8b
ollama pull nomic-embed-text

# Test any agent
python agents/top_k_retrieval.py "How does authentication work?" --repo /path/to/repo

# Test all agents
python test_agents.py
```

## Core Agents

### 1. Top-K Retrieval (`agents/top_k_retrieval.py`)
**Strategy**: Vector similarity search with top-K most relevant documents
**Speed**: Fast (15-30s)
**Best for**: General questions, quick analysis

```bash
python agents/top_k_retrieval.py "How does authentication work?" --repo /path/to/repo
python agents/top_k_retrieval.py --help
```

Options:
- `question` - Question to ask (required)
- `--repo PATH` - Repository to analyze  
- `--privacy` - Enable privacy mode

### 2. Graph-Based Retrieval (`agents/graph_based_retrieval.py`)
**Strategy**: Knowledge graph of code entities and relationships
**Speed**: Very Fast (5-15s) 
**Best for**: Questions about specific classes, functions, or relationships

```bash
# Automatically builds graph if needed
python agents/graph_based_retrieval.py "Explain the LoginViewModel class" --repo /path/to/repo

# Build graph manually first
python agents/graph_based_retrieval.py --build-graph --repo /path/to/repo
python agents/graph_based_retrieval.py --help
```

Options:
- `question` - Question to ask
- `--build-graph` - Build knowledge graph only
- `--repo PATH` - Repository to analyze

Features semantic understanding: "authentication" automatically maps to "login" functions.

### 3. Iterate and Synthesize (`agents/iterate_and_synthesize.py`)
**Strategy**: MapReduce approach - analyzes every file individually then synthesizes  
**Speed**: Slow (2-5 minutes)
**Best for**: Comprehensive analysis, architectural overviews

```bash
python agents/iterate_and_synthesize.py "Provide complete architectural analysis" --repo /path/to/repo
python agents/iterate_and_synthesize.py --help
```

Options:
- `question` - Question to ask (required)
- `--repo PATH` - Repository to analyze

**Note**: This agent is thorough but slow - it processes every file in the repository.

### 4. Multi-Representation (`agents/multi_representation.py`)
**Strategy**: Multiple document representations with adaptive strategy selection
**Speed**: Variable (30s-3min depending on strategy)
**Best for**: Complex questions requiring different analysis approaches

```bash
# Automatic strategy selection
python agents/multi_representation.py "How is user data managed?" --repo /path/to/repo

# Manual strategy override  
python agents/multi_representation.py "System overview" --strategy broad --repo /path/to/repo
python agents/multi_representation.py --help
```

Options:
- `question` - Question to ask
- `--strategy {broad,specific}` - Force specific strategy (auto-detected if not specified)
- `--build-representations` - Build representations only
- `--repo PATH` - Repository to analyze

## Advanced Features

### Hierarchical Coordinator (`hierarchical/coordinator.py`)
**Purpose**: Intelligent question routing to specialized domain experts
**Best for**: Complex questions requiring multiple perspectives

```bash
python hierarchical/coordinator.py "How does the authentication system work?" --repo /path/to/repo
python hierarchical/coordinator.py --help
```

### Workspace Manager (`intelligence/workspace_manager.py`)
**Purpose**: Repository analysis and multi-repo workspace management

```bash
# Analyze repository
python intelligence/workspace_manager.py analyze /path/to/repo

# Create multi-repo workspace
python intelligence/workspace_manager.py create-workspace my_project /repo1 /repo2

# Generate report
python intelligence/workspace_manager.py report /path/to/repo --output report.md

# See all commands
python intelligence/workspace_manager.py --help
```

## Testing & Evaluation

### Agent Testing (`test_agents.py`)
Tests all agents with standardized questions and provides performance metrics.

```bash
# Test all agents
python test_agents.py

# Test specific agent
python test_agents.py --agent top_k

# Custom repository
python test_agents.py --repo /path/to/repo

# Help
python test_agents.py --help
```

Provides comprehensive performance reports with success rates, response times, and recommendations.

### Model Evaluation (`model_evaluation.py`)
Compares different Ollama models for code analysis performance.

```bash
# Test all available models
python model_evaluation.py

# Test specific model
python model_evaluation.py --model deepseek-r1:8b

# Quick test (fewer questions)
python model_evaluation.py --quick

# Help
python model_evaluation.py --help
```

Evaluates models on success rate, response quality, and speed.

## Configuration

Edit `config.yaml` to customize:

```yaml
# LLM Provider
llm:
  provider: "ollama"  # ollama, openai, claude, gemini, wca
  models:
    ollama: "granite3.2:8b"
    openai: "gpt-4"
  temperature: 0.1

# Embeddings (always local)
embeddings:
  model: "nomic-embed-text"

# Privacy settings
privacy:
  enable: false
  mode: "fast"  # fast, strict, air-gapped
```

For cloud APIs, create `.env` file:
```bash
OPENAI_API_KEY=your_key_here
# or ANTHROPIC_API_KEY, GOOGLE_API_KEY, WCA_API_KEY
```

## Repository Specification

Three ways to specify repositories:

```bash
# 1. Command line (recommended)
--repo /path/to/repository

# 2. Environment variable
export REPO_PATH=/path/to/repository

# 3. Config file (edit config.yaml)
repository:
  local_path: "/path/to/repository"
```

## Agent Selection Guide

| Question Type | Recommended Agent | Why |
|---------------|-------------------|-----|
| "How does X work?" | Graph-Based | Fast, focuses on specific entities |
| "What's the architecture?" | Iterate & Synthesize | Comprehensive, sees whole system |
| "Quick overview of Y?" | Top-K Retrieval | Fast baseline approach |
| "Complex analysis?" | Multi-Representation | Adapts strategy to complexity |
| "Multi-domain question?" | Hierarchical | Routes to appropriate specialists |

## Performance Characteristics

From test results:
- **Graph-Based**: Fastest (7s avg), good for targeted questions
- **Top-K**: Moderate speed (29s avg), reliable baseline  
- **Iterate & Synthesize**: Slowest (134s avg) but most comprehensive
- **Multi-Representation**: Variable based on strategy selection
- **Hierarchical**: Fast coordination (11s avg), intelligent routing

## Troubleshooting

**Clear cache between repositories:**
```bash
rm -rf vector_store/ *.pkl *.gpickle
```

**Check Ollama models:**
```bash
ollama ls
ollama pull granite3.2:8b  # if missing
```

**Get help for any command:**
```bash
python <script> --help
```

## Architecture

```
agents/           # Core RAG implementations
hierarchical/     # Multi-agent orchestration  
intelligence/     # Repository analysis tools
shared/           # Common functionality
docs/             # Documentation
config.yaml       # System configuration
test_agents.py    # Comprehensive testing
model_evaluation.py  # Model comparison
```

This system transforms code repositories into queryable knowledge bases using state-of-the-art RAG techniques, making codebases more accessible and understandable through natural language interaction.