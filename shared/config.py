"""
Configuration utilities and model management.
"""

import subprocess
import sys
import os
import yaml
import re
from dotenv import load_dotenv


def load_config():
    """Load configuration from config.yaml with environment variable support.

    Supports ${ENV_VAR} and ${ENV_VAR:default_value} syntax in config.yaml values.
    Examples:
        - ollama_endpoint: ${OLLAMA_ENDPOINT}
        - provider: ${LLM_PROVIDER:ollama}
    """
    # Load environment variables from .env file (if it exists)
    load_dotenv(override=False)  # Don't override existing env vars

    with open("config.yaml", "r") as f:
        config_text = f.read()

    # Replace ${VAR} and ${VAR:default} with environment variable values
    def replace_env_vars(match):
        var_spec = match.group(1)
        if ':' in var_spec:
            var_name, default_value = var_spec.split(':', 1)
            return os.getenv(var_name, default_value)
        else:
            var_name = var_spec
            return os.getenv(var_name, match.group(0))  # Return original if not found

    config_text = re.sub(r'\$\{([^}]+)\}', replace_env_vars, config_text)

    return yaml.safe_load(config_text)


def pull_ollama_model(model_name):
    """Check for and pull Ollama model if needed (local Ollama only).

    For remote Ollama instances, assume models are already available.
    """
    try:
        print(f"Checking for Ollama model: {model_name}...")
        subprocess.run(["ollama", "show", model_name], check=True, capture_output=True, text=True)
        print(f"Model '{model_name}' is already available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ollama command fails, assume remote Ollama and model is available
        print(f"⚠️  Cannot verify model '{model_name}' (using remote Ollama?).")
        print(f"   Assuming '{model_name}' is available on the remote instance.")
        print(f"   If not, please pull the model manually on the remote Ollama server.")


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