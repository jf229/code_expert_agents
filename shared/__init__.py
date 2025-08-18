"""Shared functionality for all RAG prototypes."""

# Import main classes for easy access
from .wca_service import WCAService
from .response_generators import ResponseGenerator, UnifiedResponseGenerator
from .config import load_config, setup_and_pull_models, pull_ollama_model
from .llm_providers import get_llm_provider
from .data_ingestion import main as data_ingestion_main

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