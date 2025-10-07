# Agents Overview

Short, practical reference for the code-expert agents and when to use each.

## Core Agents

- Top‑K Retrieval (`agents/top_k_retrieval.py`)
  - Strategy: Vector similarity + parent/child chunks.
  - Best for: Quick, specific questions; baseline analyses.
  - Pros: Fast, simple, dependable. Cons: Limited global context.
  - Run: `python agents/top_k_retrieval.py "<question>" --repo <path>`

- Graph‑Based Retrieval (`agents/graph_based_retrieval.py`)
  - Strategy: Code knowledge graph (files, classes, functions) + semantic matching.
  - Best for: Entity/relationship questions (e.g., a class or function and its links).
  - Pros: Very fast, precise entity focus. Cons: Needs a built graph for best results.
  - Run: `python agents/graph_based_retrieval.py "<question>" --repo <path>`
  - Build graph (optional): `python agents/graph_based_retrieval.py --build-graph --repo <path>`

- Iterate & Synthesize (`agents/iterate_and_synthesize.py`)
  - Strategy: Map‑reduce over every file → final synthesis.
  - Best for: Whole‑repo architectural overviews and exhaustive reads.
  - Pros: Most comprehensive. Cons: Slowest.
  - Run: `python agents/iterate_and_synthesize.py "<question>" --repo <path>`

- Multi‑Representation (`agents/multi_representation.py`)
  - Strategy: Multiple representations (summaries + hypothetical Qs) with adaptive retrieval.
  - Best for: Mixed/complex questions; balances breadth vs. focus.
  - Pros: Flexible, adaptive. Cons: Initial representation build is slower.
  - Build reps (once): `python agents/multi_representation.py --build-representations --repo <path>`
  - Run: `python agents/multi_representation.py "<question>" [--strategy broad|specific] --repo <path>`

## Orchestration

- Hierarchical Coordinator (`hierarchical/coordinator.py`)
  - Routes questions to the best agent(s) based on type, domain, and complexity.
  - Supports single, sequential, parallel, and collaborative patterns.
  - Run: `python hierarchical/coordinator.py "<question>" --repo <path>`

- Domain Specialists (`hierarchical/specialists.py`)
  - Backend (Graph‑Based), Frontend (Top‑K), Data (Graph‑Based), Infrastructure (Iterate & Synthesize), Architecture (Multi‑Representation).

## Shared Services (used by agents)

- Data Ingestion (`shared/data_ingestion.py`): Intelligent file loading; saves raw docs to storage.
- Storage (`config.yaml > storage`):
  - `raw_docs` (pickled documents), `vector_store` (Chroma), `code_graph` (pickle), `multi_representations` (pickle).
- Privacy (`intelligence/privacy_manager.py`): Optional sanitization/auditing when sending context to external LLMs.
- LLM Providers (`shared/llm_providers.py`): Default `ollama`; supports OpenAI, Claude, Gemini, and WCA.
- Unified Responses (`shared/response_generators.py`): Consistent formatting and source listing.

## Quick Picks

- “What’s the architecture?” → Iterate & Synthesize
- “Explain class/function X.” → Graph‑Based Retrieval
- “How does auth/data flow work?” → Top‑K or Multi‑Representation (broad)
- “Cross‑domain, complex question.” → Hierarchical Coordinator

## Notes

- Specify repo via `--repo`, `REPO_PATH`, or `config.yaml`.
- Pull local models first for best performance (see README.md Quick Start).
