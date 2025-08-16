# main.py
import sys
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
    parser = argparse.ArgumentParser(description="RAG-based Code Expert Agent (Multi-Representation)")
    parser.add_argument("question", type=str, help="The question to ask the agent.")
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=['specific', 'broad'], 
        default='specific', 
        help="The retrieval strategy to use."
    )
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    llm_model = config["llm"]["ollama_model"]
    embedding_model = config["embeddings"]["model"]

    pull_ollama_model(llm_model)
    pull_ollama_model(embedding_model)

    print(f"--- RAG-based Code Expert Agent (Multi-Representation) ---")
    print(f"Strategy: {args.strategy}")
    
    # This prototype has its own data pipeline, so we don't call the central one.
    # The user is expected to run the builder scripts manually as per the README.

    print("--- Agent Orchestration ---")
    qa = agent_orchestration_main(strategy=args.strategy)
    
    if qa is None:
        print("Agent orchestration failed. Exiting.")
        sys.exit(1)

    print("--- Reasoning and Response Generation ---")
    reasoning_response_generation_main(qa, args.question)
    
    print("--- System Finished ---")

if __name__ == "__main__":
    main()
