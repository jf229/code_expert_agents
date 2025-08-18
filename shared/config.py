"""
Configuration utilities and model management.
"""

import subprocess
import sys
import yaml


def pull_ollama_model(model_name):
    """Check for and pull Ollama model if needed."""
    try:
        print(f"Checking for Ollama model: {model_name}...")
        subprocess.run(["ollama", "show", model_name], check=True, capture_output=True, text=True)
        print(f"Model '{model_name}' is already available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Model '{model_name}' not found. Pulling from Ollama...")
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"Model '{model_name}' pulled successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Failed to pull Ollama model '{model_name}'.")
            sys.exit(1)


def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def setup_and_pull_models(config):
    """Pull required models based on configuration."""
    llm_provider = config["llm"]["provider"]
    embedding_model = config["embeddings"]["model"]
    
    # Always pull embedding model (always uses Ollama)
    pull_ollama_model(embedding_model)
    
    # Pull LLM model only if using Ollama
    if llm_provider == "ollama":
        llm_model = config["llm"]["models"]["ollama"]
        pull_ollama_model(llm_model)