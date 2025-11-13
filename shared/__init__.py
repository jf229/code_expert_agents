"""Shared functionality for all RAG prototypes.

Note: Lazy-load heavy modules (e.g., data_ingestion) to avoid import-time deps
in lightweight tests.
"""

# Import main classes for easy access
from .wca_service import WCAService
from .response_generators import ResponseGenerator, UnifiedResponseGenerator
from .config import load_config, setup_and_pull_models, pull_ollama_model
from .llm_providers import get_llm_provider


def data_ingestion_main(*args, **kwargs):
    """Lazy proxy to shared.data_ingestion.main to avoid import-time deps."""
    from .data_ingestion import main as _main
    return _main(*args, **kwargs)


__all__ = [
    'WCAService',
    'ResponseGenerator', 
    'UnifiedResponseGenerator',
    'load_config',
    'setup_and_pull_models', 
    'pull_ollama_model',
    'get_llm_provider',
    'data_ingestion_main'
]
