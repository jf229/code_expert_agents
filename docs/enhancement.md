# Repository-Specific Code Intelligence Enhancement

## Core Vision

Create a persistent, repository-specific code intelligence system where each repo maintains its own knowledge base that evolves with the codebase.

## Repository-Specific Storage Strategy

### Current Storage (Temporary)
```
current_dir/
├── vector_store/
├── raw_documents.pkl
├── code_graph.gpickle
└── multi_representations.pkl
```

### Enhanced Storage (Persistent)
```
target_repo/
├── .code_expert/
│   ├── vector_store/
│   ├── raw_documents.pkl
│   ├── code_graph.gpickle
│   ├── multi_representations.pkl
│   ├── metadata.json  # versioning, last updated, etc.
│   └── config.yaml    # repo-specific settings
├── src/
└── README.md
```

## Command-Line Utility Vision

```bash
# Install code expert for a repository
code-expert init /path/to/repo

# Ask questions (automatically uses repo's knowledge base)
code-expert ask "How does authentication work?" --repo /path/to/repo

# Update knowledge base when code changes
code-expert refresh --repo /path/to/repo

# Choose specific agent strategy
code-expert ask "Explain UserController" --agent graph --repo /path/to/repo

# Global installation
code-expert ask "System architecture overview" --agent iterate
```

## Benefits

1. **Persistent Knowledge**: No more rebuilding indexes every time
2. **Repository Evolution**: Knowledge base grows with the codebase
3. **Team Sharing**: `.code_expert/` can be gitignored or shared with team
4. **Performance**: Instant responses after initial setup
5. **Version Tracking**: Track when knowledge base was last updated vs code changes

## Implementation Approach

We could begin with:

1. Modify storage paths to use target repo
2. Add metadata tracking for incremental updates
3. Create a unified CLI interface

## Current System Test

Right now we can test this concept by cloning this tool into each repository and running with `--repo=../` to analyze the parent directory, while working toward storing the knowledge base in the target repository's `.code_expert/` directory.