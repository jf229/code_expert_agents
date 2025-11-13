"""
BaseAgent: common template for all code expert agents.

Provides a consistent lifecycle:
- prepare(question): optional setup (e.g., ensure storages)
- retrieve(question) -> List[Document]: must be implemented by subclasses
- analyze(question, docs): default unified response generation
- run(question): orchestrates the above
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional

from . import load_config, UnifiedResponseGenerator


class BaseAgent(ABC):
    """Abstract base class for agents with a unified run template."""

    def __init__(self, name: str = "RAG Agent"):
        self.name = name
        self.config = load_config()
        # Unified response generator handles provider selection (Ollama/WCA/etc.)
        self.unified = UnifiedResponseGenerator(self.config, prototype_name=name)

    # Optional helpers for subclasses to override when relevant
    def ensure_raw_documents(self) -> None:
        pass

    def ensure_vector_store(self) -> None:
        pass

    def ensure_graph(self) -> None:
        pass

    def ensure_representations(self) -> None:
        pass

    def prepare(self, question: str) -> None:
        """Optional preparation step before retrieval."""
        pass

    @abstractmethod
    def retrieve(self, question: str) -> List[Any]:
        """Return a list of retrieved documents for the question."""
        raise NotImplementedError

    def analyze(self, question: str, documents: List[Any]) -> Optional[dict]:
        """Default analysis: generate response with retrieved context."""
        if not documents:
            print("No relevant documents found for the query.")
            return None
        return self.unified.generate_response_with_context(question, documents)

    def run(self, question: str):
        print(f"--- {self.name} ---")
        # 1) Prepare (subclass may ensure storages)
        self.prepare(question)
        # 2) Retrieve
        docs = self.retrieve(question)
        # 3) Analyze and output
        result = self.analyze(question, docs)
        print("--- System Finished ---")
        return result

