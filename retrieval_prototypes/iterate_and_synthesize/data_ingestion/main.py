# data_ingestion/main.py
import os
import yaml
import pickle
from git import Repo
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Expanded list of common programming and markup language file extensions
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

def get_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def get_repo(config):
    repo_path = "repo"
    local_repo_path = config.get("local_repo_path")

    if local_repo_path and os.path.isdir(local_repo_path):
        print(f"Using local repository at: {local_repo_path}")
        return local_repo_path
    
    repo_url = config.get("repository_url")
    if not repo_url:
        raise ValueError("Either 'local_repo_path' or 'repository_url' must be provided in config.yaml")

    if not os.path.exists(repo_path):
        print(f"Cloning repository: {repo_url}")
        Repo.clone_from(repo_url, repo_path)
    else:
        print(f"Using existing cloned repository at: {repo_path}")
    return repo_path

def load_documents(repo_dir):
    documents = []
    print("\n--- Starting Document Loading ---")
    
    extensions = set(COMMON_LANGUAGE_EXTENSIONS.values())
    
    for ext in extensions:
        glob_pattern = f"**/*{ext}"
        try:
            loader = DirectoryLoader(
                repo_dir,
                glob=glob_pattern,
                loader_cls=TextLoader,
                show_progress=False,
                use_multithreading=True,
                silent_errors=True
            )
            loaded_docs = loader.load()
            
            if loaded_docs:
                print(f"Found {len(loaded_docs)} files with extension '{ext}':")
                for doc in loaded_docs:
                    relative_path = os.path.relpath(doc.metadata['source'], repo_dir)
                    print(f"  - Loaded: {relative_path}")
                documents.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading files with pattern {glob_pattern}: {e}")
            continue
            
    print("--- Finished Document Loading ---\n")
    return documents

def main():
    print("--- Data Ingestion Component ---")
    config = get_config()
    repo_dir = get_repo(config)

    documents = load_documents(repo_dir)
    
    if not documents:
        print("No documents were loaded. Please check the repository and file extensions.")
        return

    # Save the raw, un-chunked documents
    output_path = "data_ingestion/raw_documents.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"Raw documents saved to: {output_path}")
    print("--- Data Ingestion Complete---\n")

if __name__ == "__main__":
    main()