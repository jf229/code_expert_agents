# Enhancement: LLM Processing Cache for Iterate & Synthesize Agent

## Problem Statement

The Iterate & Synthesize agent is extremely thorough but prohibitively slow because it processes every file through the LLM on each run. For a typical repository with 100+ files, this can take 10-50 minutes per analysis, making it impractical for iterative development workflows.

## Current Behavior

```python
# In _process_all_files_ollama() and _process_all_files_wca()
for doc in tqdm(documents, desc="Processing files"):
    # This LLM call takes 5-30 seconds per file
    summary_response = summary_chain.run(file_content=file_content)
    summaries.append(summary_text)
```

**Performance Impact:**
- 100 files × 10 seconds/file = ~17 minutes
- No incremental processing - full re-analysis every time
- Makes the agent unsuitable for iterative use

## Proposed Solution

Implement intelligent caching that stores LLM-generated summaries based on file content hashes, enabling:

1. **File-level caching**: Cache individual file summaries
2. **Incremental processing**: Only process changed files
3. **Repository-level synthesis caching**: Cache final analysis based on combined file states

## Technical Implementation

### 1. Cache Manager (New Component)

```python
# shared/cache_manager.py
class LLMCacheManager:
    def __init__(self, cache_dir=".llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.file_cache = self._load_cache("file_summaries.json")
        self.synthesis_cache = self._load_cache("synthesis_results.json")
    
    def get_file_hash(self, file_path: str, content: str) -> str:
        """Generate hash for file content + metadata"""
        combined = f"{file_path}:{len(content)}:{hashlib.md5(content.encode()).hexdigest()}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get_cached_summary(self, file_path: str, content: str) -> Optional[str]:
        """Retrieve cached file summary if available"""
        cache_key = self.get_file_hash(file_path, content)
        return self.file_cache.get(cache_key)
    
    def cache_summary(self, file_path: str, content: str, summary: str):
        """Store file summary in cache"""
        cache_key = self.get_file_hash(file_path, content)
        self.file_cache[cache_key] = {
            "summary": summary,
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content)
        }
        self._save_cache("file_summaries.json", self.file_cache)
```

### 2. Integration into Processing Loop

```python
# Modified _process_all_files_ollama()
def _process_all_files_ollama(self, documents, llm):
    cache_manager = LLMCacheManager()
    summaries = []
    cache_hits = 0
    
    for doc in tqdm(documents, desc="Processing files"):
        file_path = doc.metadata.get('source', 'unknown')
        file_content = doc.page_content
        
        # Check cache first
        cached_summary = cache_manager.get_cached_summary(file_path, file_content)
        if cached_summary:
            summary_response = cached_summary["summary"]
            cache_hits += 1
        else:
            # Process with LLM and cache result
            summary_response = summary_chain.run(file_content=file_content)
            cache_manager.cache_summary(file_path, file_content, summary_response)
        
        summary_text = f"**{os.path.basename(file_path)}:**\n{summary_response}\n"
        summaries.append(summary_text)
    
    print(f"Cache hits: {cache_hits}/{len(documents)} files")
```

### 3. Synthesis-Level Caching

```python
def get_synthesis_hash(self, all_summaries: str) -> str:
    """Generate hash for final synthesis input"""
    return hashlib.sha256(all_summaries.encode()).hexdigest()[:16]

def get_cached_synthesis(self, all_summaries: str) -> Optional[str]:
    """Check if synthesis result is cached"""
    synthesis_key = self.get_synthesis_hash(all_summaries)
    return self.synthesis_cache.get(synthesis_key)

def cache_synthesis(self, all_summaries: str, final_analysis: str):
    """Cache the final synthesis result"""
    synthesis_key = self.get_synthesis_hash(all_summaries)
    self.synthesis_cache[synthesis_key] = {
        "analysis": final_analysis,
        "timestamp": datetime.now().isoformat(),
        "summary_count": len(all_summaries.split("**")) - 1
    }
```

## Performance Impact

### Before Caching
- **First run**: 100 files × 10 seconds = ~17 minutes
- **Subsequent runs**: 100 files × 10 seconds = ~17 minutes (no improvement)
- **Small changes**: Still 17 minutes for 1-file change

### After Caching
- **First run**: 100 files × 10 seconds = ~17 minutes (cache population)
- **Subsequent runs (no changes)**: ~5 seconds (all cache hits)
- **Incremental changes**: Changed files × 10 seconds + ~5 seconds cache reads
- **Speed improvement**: **3000-30000x faster** for cache hits

### Real-World Scenarios
1. **Daily code reviews**: 2-3 changed files = ~30 seconds vs 17 minutes
2. **Architecture updates**: Full repo scan in seconds after initial run
3. **CI/CD integration**: Fast enough for automated analysis

## Implementation Scope

**Files to modify:**
1. `agents/iterate_and_synthesize.py` - integrate cache checks
2. `shared/cache_manager.py` - new cache management component
3. `config.yaml` - add cache configuration options

**Estimated effort:** ~100 lines of code across 3 files

**Configuration additions:**
```yaml
caching:
  enabled: true
  cache_dir: ".llm_cache"
  max_cache_age_days: 30
  cache_compression: true
```

## Cache Management Features

1. **Automatic cleanup**: Remove old cache entries
2. **Cache statistics**: Show hit/miss ratios
3. **Manual cache control**: Clear cache commands
4. **Cross-session persistence**: Cache survives restarts
5. **Content-based invalidation**: Automatic cache invalidation when files change

## Future Enhancements

1. **Distributed caching**: Share cache across team members
2. **Semantic similarity**: Cache hits for similar but not identical content
3. **Partial synthesis**: Cache intermediate synthesis steps
4. **Background pre-processing**: Update cache for changed files in background

## Risk Mitigation

1. **Cache corruption**: Validate cache integrity on load
2. **Disk space**: Implement cache size limits and cleanup
3. **Stale results**: Include file modification timestamps in cache keys
4. **Version compatibility**: Cache format versioning for future changes

This enhancement would transform the Iterate & Synthesize agent from a "once-per-project" tool into a practical iterative development companion.