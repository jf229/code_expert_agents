# Hierarchical Multi-Agent System

## Overview

The Hierarchical Multi-Agent System implements an intelligent router that automatically selects the best agent and strategy based on question analysis and available storage systems. It preserves all existing functionality while adding smart orchestration.

## Architecture

```
Master Coordinator
├── Question Classifier (analyzes intent, complexity, domain)
├── Storage Analyzer (checks available storage systems)
├── Agent Router (selects optimal agent combination)
└── Orchestration Engine (executes single/multi-agent workflows)
```

## Key Components

### 1. Hierarchical Coordinator (`hierarchical_coordinator.py`)
- **Master orchestrator** that routes questions intelligently
- Analyzes questions for type (what/how/why/where), domain, and complexity
- Routes to optimal agent based on question characteristics
- Supports single, sequential, parallel, and collaborative execution patterns

### 2. Domain Specialists (`domain_specialists.py`)
- **Backend Specialist**: APIs, business logic, services (uses Graph-based agent)
- **Frontend Specialist**: UI components, interactions (uses Top-K agent)
- **Data Specialist**: Models, databases, data flow (uses Graph-based agent)
- **Infrastructure Specialist**: Deployment, config, DevOps (uses Iterate & Synthesize)
- **Architecture Specialist**: System design, patterns (uses Multi-representation)

### 3. Enhanced Question Analyzer (`enhanced_question_analyzer.py`)
- Analyzes question complexity across multiple dimensions
- Checks availability of storage systems (vector store, graph, multi-representations)
- Recommends optimal storage and agent combinations
- Provides storage optimization suggestions

## Usage

### Basic Hierarchical Routing
```bash
# Let the system automatically choose the best approach
python hierarchical_coordinator.py "what is the UserService class?" --repo /path/to/repo

# System will:
# 1. Classify as "what" + "backend" + "simple"
# 2. Route to Backend Specialist using Graph-based agent
# 3. Execute with backend-specific prompting
```

### Direct Domain Specialist Usage
```bash
# Use specific domain specialist directly
python domain_specialists.py backend "how does authentication work?" --repo /path/to/repo
python domain_specialists.py frontend "what is the LoginComponent?" --repo /path/to/repo
python domain_specialists.py data "explain the user data model" --repo /path/to/repo
```

### Storage Analysis and Optimization
```bash
# Analyze question and get storage recommendations
python enhanced_question_analyzer.py "how does the payment flow work?" --repo /path/to/repo

# Get detailed analysis
python enhanced_question_analyzer.py "explain the architecture" --detailed --repo /path/to/repo
```

## Routing Examples

### Simple Entity Questions
```
Q: "What is the UserService class?"
├── Classification: What + Backend + Simple
├── Route: Backend Specialist (Graph-based retrieval)
└── Execution: Single agent, entity-focused analysis
```

### Complex Process Questions
```
Q: "How does the entire checkout process work from cart to payment?"
├── Classification: How + Cross-domain + Complex  
├── Route: Multi-agent collaborative workflow
├── Agents: Architecture + Backend + Frontend + Data specialists
└── Execution: Collaborative analysis with synthesis
```

### Cross-Domain Questions
```
Q: "Why does frontend validation duplicate backend validation?"
├── Classification: Why + Frontend + Backend + Medium
├── Route: Sequential analysis with synthesis
├── Agents: Frontend → Backend → Architecture specialists
└── Execution: Sequential with architectural reasoning
```

## Storage Optimization

The system leverages existing storage optimally:

### Vector Store (`vector_store/`)
- **Best for**: Fast entity retrieval, specific component questions
- **Used by**: Top-K agent, Frontend specialist
- **Created by**: Running any agent

### Code Graph (`code_graph.gpickle`)
- **Best for**: Relationship analysis, entity dependencies
- **Used by**: Graph-based agent, Backend/Data specialists
- **Created by**: `python graph_based_retrieval.py --build-graph`

### Multi-Representations (`multi_representations.pkl`)
- **Best for**: Adaptive strategies, complex analysis
- **Used by**: Multi-representation agent, Architecture specialist
- **Created by**: `python multi_representation.py --build-representations`

### Raw Documents (`raw_documents.pkl`)
- **Best for**: Comprehensive analysis, system-wide questions
- **Used by**: Iterate & Synthesize agent, Infrastructure specialist
- **Created by**: Data ingestion (automatic)

## Intelligent Features

### Automatic Question Classification
- **What questions** → Entity-focused retrieval (Graph-based)
- **How questions** → Process-focused analysis (Multi-agent if complex)
- **Why questions** → Rationale-focused synthesis (Iterate & Synthesize)
- **Where questions** → Location-focused search (Top-K with escalation)

### Domain-Aware Routing
- **Backend questions** → Graph-based for precise entity relationships
- **Frontend questions** → Top-K for quick component retrieval
- **Data questions** → Graph-based for relationship analysis
- **Infrastructure questions** → Iterate & Synthesize for comprehensive coverage
- **Cross-domain questions** → Multi-agent collaborative workflows

### Complexity-Based Orchestration
- **Simple questions** → Single agent, focused analysis
- **Medium questions** → Enhanced prompting, possible secondary agent
- **Complex questions** → Multi-agent workflows with synthesis

## Benefits

### For Users
- **No agent selection needed** - system chooses optimal approach
- **Consistent quality** - domain expertise applied automatically
- **Comprehensive coverage** - complex questions get multi-agent treatment
- **Natural language** - ask questions naturally, get intelligent routing

### For Developers
- **Preserves existing agents** - no changes to current functionality
- **Additive architecture** - hierarchical system is an overlay
- **Storage optimization** - leverages existing storage systems efficiently
- **Extensible design** - easy to add new specialists or routing rules

## Configuration

The system uses the existing `config.yaml` with no additional configuration required. All routing and orchestration is automatic based on question analysis.

## Future Enhancements

### Planned Features
- **Learning-based routing** - improve routing based on success metrics
- **Context-aware conversations** - maintain context across questions
- **Confidence-based escalation** - escalate to more powerful agents when confidence is low
- **Cross-agent knowledge sharing** - agents share findings for better synthesis

### Extension Points
- **New domain specialists** - add security, mobile, ML specialists
- **Custom routing rules** - project-specific routing patterns
- **External integrations** - integrate with documentation, issue trackers
- **Performance metrics** - track and optimize agent performance

## Testing

Test the hierarchical system with different question types:

```bash
# Entity questions (should route to Graph-based)
python hierarchical_coordinator.py "what is the DatabaseConnection class?" --repo /path/to/repo

# Process questions (should route to multi-agent for complex ones)
python hierarchical_coordinator.py "how does user registration work end-to-end?" --repo /path/to/repo

# Architecture questions (should route to Multi-representation)
python hierarchical_coordinator.py "explain the overall system architecture" --repo /path/to/repo

# Infrastructure questions (should route to Iterate & Synthesize)
python hierarchical_coordinator.py "how is the application deployed?" --repo /path/to/repo
```

The hierarchical system provides intelligent routing while preserving all existing functionality, making the code expert system more accessible and effective for users.