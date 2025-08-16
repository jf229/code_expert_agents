# main.py
import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import subprocess
import yaml
from data_ingestion import main as data_ingestion_main
from agent_orchestration.main import main as agent_orchestration_main
from reasoning_response_generation.main import main as reasoning_response_generation_main

def pull_ollama_model(model_name):
    # ... (same as before)
    try:
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

def main():
    parser = argparse.ArgumentParser(description="RAG-based Code Expert Agent (Graph-Based Retrieval)")
    parser.add_argument("question", type=str, help="The question to ask the agent.")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    llm_model = config["llm"]["ollama_model"]
    embedding_model = config["embeddings"]["model"]

    pull_ollama_model(llm_model)
    pull_ollama_model(embedding_model)

    print("--- RAG-based Code Expert Agent (Graph-Based Retrieval) ---")
    
    print("--- Ensuring repository is available ---")
    data_ingestion_main()

    print("--- Agent Orchestration ---")
    qa = agent_orchestration_main()
    
    if qa is None:
        print("Agent orchestration failed. Exiting.")
        sys.exit(1)

    print("--- Reasoning and Response Generation ---")
    reasoning_response_generation_main(qa, args.question)
    
    print("--- System Finished ---")

if __name__ == "__main__":
    main()