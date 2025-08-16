# data_ingestion.py
import os
import yaml
import pickle
from git import Repo
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
import sys

# This is now the central list of supported languages
COMMON_LANGUAGE_EXTENSIONS = {
    "Assembly": ".asm", "Bash": ".sh", "C": ".c", "C++": ".cpp", "C#": ".cs",
    "Clojure": ".clj", "CMake": ".cmake", "CoffeeScript": ".coffee", "CSS": ".css",
    "Dart": ".dart", "Dockerfile": ".dockerfile", "Elixir": ".ex", "Erlang": ".erl",
    "Go": ".go", "Groovy": ".groovy", "Haskell": ".hs", "HTML": ".html", "Java": ".java",
    "JavaScript": ".js", "JSON": ".json", "JSX": ".jsx", "Julia": ".jl",
    "Jupyter Notebook": ".ipynb", "Kotlin": ".kt", "LaTeX": ".tex", "Lisp": ".lisp",
    "Lua": ".lua", "Makefile": ".mk", "Markdown": ".md", "MATLAB": ".m",
    "Objective-C": ".m", "OCaml": ".ml", "Perl": ".pl", "PHP": ".php",
    "PowerShell": ".ps1", "Python": ".py", "R": ".r", "Ruby": ".rb", "Rust": ".rs",
    "Scala": ".scala", "SCSS": ".scss", "SQL": ".sql", "Swift": ".swift",
    "Svelte": ".svelte", "TeX": ".tex", "TOML": ".toml", "TypeScript": ".ts",
    "TSX": ".tsx", "Vue": ".vue", "XML": ".xml", "YAML": ".yaml", "YML": ".yml"
}

def get_repo(config):
    repo_config = config.get("repository", {})
    local_path = repo_config.get("local_path")
    
    if local_path and os.path.isdir(local_path):
        print(f"Using local repository at: {local_path}")
        return local_path
    
    remote_url = repo_config.get("remote_url")
    if remote_url:
        target_path = "repo" # A default clone directory
        if not os.path.exists(target_path):
            print(f"Cloning repository: {remote_url}")
            Repo.clone_from(remote_url, target_path)
        else:
            print(f"Using existing cloned repository at: {target_path}")
        return target_path
        
    raise ValueError("No valid 'local_path' or 'remote_url' provided in config.yaml")

def load_documents(repo_dir):
    documents = []
    print("\n--- Starting Document Loading ---")
    
    extensions = set(COMMON_LANGUAGE_EXTENSIONS.values())
    
    all_files = [os.path.join(root, file) for root, _, files in os.walk(repo_dir) for file in files]
    source_files = [f for f in all_files if os.path.splitext(f)[1] in extensions]
    
    print(f"Found {len(source_files)} source files to load.")

    for file_path in tqdm(source_files, desc="Loading Documents"):
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
            
    print("--- Finished Document Loading ---\n")
    return documents

def main():
    print("--- Central Data Ingestion ---")
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: Root config.yaml not found.")
        sys.exit(1)

    repo_dir = get_repo(config)
    documents = load_documents(repo_dir)
    
    if not documents:
        print("No documents were loaded. Please check the repository and file extensions.")
        return

    output_path = config["storage"]["raw_docs"]
    with open(output_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"Raw documents saved to: {output_path}")
    print("--- Data Ingestion Complete---\n")

if __name__ == "__main__":
    main()