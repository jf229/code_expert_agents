# data_ingestion.py
import os
import yaml
import pickle
import fnmatch
from pathlib import Path
from git import Repo
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
import sys

# Import repository intelligence
try:
    # Add parent directory to path to access intelligence module
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from intelligence.repository_intelligence import RepositoryIntelligence
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False
    print("Repository Intelligence not available - using basic mode")

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

def get_repo(config, repo_path=None):
    """Get repository path from config, environment, or command line argument."""
    
    # Priority: command line arg > environment variable > config file
    if repo_path:
        if os.path.isdir(repo_path):
            print(f"Using repository from argument: {repo_path}")
            return repo_path
        else:
            raise ValueError(f"Repository path does not exist: {repo_path}")
    
    # Check environment variable
    env_repo = os.environ.get("REPO_PATH")
    if env_repo and os.path.isdir(env_repo):
        print(f"Using repository from environment: {env_repo}")
        return env_repo
    
    # Fall back to config file
    repo_config = config.get("repository", {})
    local_path = repo_config.get("local_path")
    
    if local_path and os.path.isdir(local_path):
        print(f"Using repository from config: {local_path}")
        return local_path
    
    remote_url = repo_config.get("remote_url")
    if remote_url:
        target_path = "repo"
        if not os.path.exists(target_path):
            print(f"Cloning repository: {remote_url}")
            Repo.clone_from(remote_url, target_path)
        else:
            print(f"Using existing cloned repository at: {target_path}")
        return target_path
        
    raise ValueError("No repository specified. Provide via:\n"
                   "  - Command line argument\n" 
                   "  - REPO_PATH environment variable\n"
                   "  - config.yaml repository.local_path")

def load_documents(repo_dir, config=None):
    """Load documents with intelligent filtering and optimization."""
    documents = []
    print("\n--- Starting Intelligent Document Loading ---")
    
    # Get intelligent metadata if available
    metadata = config.get('_intelligent_metadata') if config else None
    
    if metadata and INTELLIGENCE_AVAILABLE:
        print("🧠 Using intelligent filtering...")
        
        # Use language-specific extensions if available
        if metadata.primary_languages:
            smart_extensions = set()
            for lang in metadata.primary_languages:
                if lang in COMMON_LANGUAGE_EXTENSIONS:
                    ext = COMMON_LANGUAGE_EXTENSIONS[lang]
                    if isinstance(ext, str):
                        smart_extensions.add(ext)
                    else:
                        smart_extensions.update(ext)
            
            # Add common config/doc files
            smart_extensions.update(['.json', '.yaml', '.yml', '.md', '.txt', '.toml'])
            extensions = smart_extensions
            print(f"   - Focused on {len(extensions)} language-specific extensions")
        else:
            extensions = set(COMMON_LANGUAGE_EXTENSIONS.values())
            
        # Use recommended excludes
        exclude_patterns = metadata.recommended_excludes
        print(f"   - Applying {len(exclude_patterns)} intelligent exclusion patterns")
        
    else:
        print("📄 Using standard filtering...")
        extensions = set(COMMON_LANGUAGE_EXTENSIONS.values())
        exclude_patterns = {
            'node_modules', '.git', '.svn', '__pycache__', 'build', 'dist', 
            'target', '.vscode', '.idea', '*.log', '*.tmp'
        }
    
    # Walk directory with intelligent filtering
    all_files = []
    for root, dirs, files in os.walk(repo_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if not _should_exclude_directory(d, exclude_patterns)]
        
        for file in files:
            file_path = os.path.join(root, file)
            if _should_include_file(file_path, extensions, exclude_patterns):
                all_files.append(file_path)
    
    print(f"📊 Intelligent filtering results:")
    print(f"   - Found {len(all_files)} files to process")
    
    # Apply processing optimizations based on metadata
    if metadata:
        if metadata.processing_priority == "low":
            print("   - Large repository detected: Using batch processing")
            batch_size = 50
        elif metadata.processing_priority == "high":
            print("   - Small repository detected: Using optimized processing")
            batch_size = None
        else:
            batch_size = 100
    else:
        batch_size = None
    
    # Load documents with progress tracking
    if batch_size:
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i + batch_size]
            for file_path in tqdm(batch, desc=f"Loading Batch {i//batch_size + 1}"):
                _load_single_document(file_path, documents)
    else:
        for file_path in tqdm(all_files, desc="Loading Documents"):
            _load_single_document(file_path, documents)
            
    print("--- Finished Intelligent Document Loading ---\n")
    return documents


def _should_exclude_directory(dirname, exclude_patterns):
    """Check if directory should be excluded based on intelligent patterns."""
    if dirname.startswith('.'):
        return True
    
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(dirname, pattern):
            return True
    
    return False


def _should_include_file(file_path, extensions, exclude_patterns):
    """Check if file should be included based on intelligent filtering."""
    file_path_obj = Path(file_path)
    filename = file_path_obj.name
    
    # Check extension
    if file_path_obj.suffix not in extensions and filename not in extensions:
        return False
    
    # Check exclude patterns
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(str(file_path_obj), pattern):
            return False
    
    # Skip very large files (> 1MB)
    try:
        if file_path_obj.stat().st_size > 1024 * 1024:
            return False
    except (OSError, PermissionError):
        return False
    
    return True


def _load_single_document(file_path, documents):
    """Load a single document with error handling."""
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    except Exception as e:
        # Only show error for important files
        if not file_path.endswith(('.log', '.tmp', '.cache')):
            print(f"Could not load {file_path}: {e}")

def main(repo_path=None):
    print("--- Central Data Ingestion ---")
    
    # Handle command line args only if called directly
    if repo_path is None and __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="Data ingestion for RAG agents")
        parser.add_argument("--repo", help="Repository path to analyze")
        args = parser.parse_args()
        repo_path = args.repo
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: Root config.yaml not found.")
        sys.exit(1)

    repo_dir = get_repo(config, repo_path)
    documents = load_documents(repo_dir)
    
    if not documents:
        print("No documents were loaded. Please check the repository and file extensions.")
        return

    # Save raw documents
    output_path = config["storage"]["raw_docs"]
    with open(output_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"Raw documents saved to: {output_path}")
    
    # If intelligence is available, save enhanced metadata
    if '_intelligent_metadata' in config and INTELLIGENCE_AVAILABLE:
        metadata = config['_intelligent_metadata']
        metadata_path = "repository_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Repository metadata saved to: {metadata_path}")
        
        # Generate intelligence report
        ri = RepositoryIntelligence()
        report = ri.generate_report(repo_dir)
        with open("repository_intelligence_report.md", "w") as f:
            f.write(report)
        print("📊 Repository intelligence report saved to: repository_intelligence_report.md")
    
    print("--- Data Ingestion Complete ---\n")

if __name__ == "__main__":
    main()