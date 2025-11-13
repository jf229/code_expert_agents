# Test Plan - Code Expert Agents with Project-Scoped Storage

## Overview
This test plan validates the new project-scoped storage system where each analyzed repository gets its own isolated storage directory, and the QueryAgent can answer questions based on pre-computed comprehensive analysis.

## Setup

### Prerequisites
1. Clutch box Llama running at `clutchbox.local:11434`
2. `.env` file configured with:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_ENDPOINT=http://clutchbox.local:11434
   OLLAMA_MODEL=granite3.2:8b
   ```
3. Test repositories available (local paths):
   - `/Users/clutchcoder/working/bikecheck` (Swift iOS app)
   - `/Users/clutchcoder/working/rh_web` (Python web application)
   - `/Users/clutchcoder/working/ansible-lint-fixer` (Python Ansible tool)

## Test Cases

### Phase 1: Configuration & Environment Variables
**Goal**: Verify environment variable loading works correctly

- [ ] **Test 1.1**: Verify `.env` file is loaded
  - Create `.env` with test values
  - Run: `python -c "from shared import load_config; c = load_config(); print(c['llm']['provider'])"`
  - Expected: `ollama`

- [ ] **Test 1.2**: Verify OLLAMA_ENDPOINT is read from .env
  - Run: `python -c "from shared import load_config; c = load_config(); print(c['llm']['ollama_endpoint'])"`
  - Expected: `http://clutchbox.local:11434`

- [ ] **Test 1.3**: Verify OLLAMA_MODEL is read from .env
  - Run: `python -c "from shared import load_config; c = load_config(); print(c['llm']['models']['ollama'])"`
  - Expected: `granite3.2:8b`

### Phase 2: Storage Manager - Project-Scoped Paths
**Goal**: Verify storage manager creates correct project-specific directories

- [ ] **Test 2.1**: Verify project directory creation
  - Run: `python -c "from shared.storage_manager import get_project_storage_dir; print(get_project_storage_dir('/path/to/test_repo'))"`
  - Expected: `project_storage/test_repo` directory created

- [ ] **Test 2.2**: Verify multiple projects get isolated storage
  - Create two test repos: `/tmp/project_a` and `/tmp/project_b`
  - Verify paths are: `project_storage/project_a/` and `project_storage/project_b/`
  - Expected: Two separate directories

### Phase 3: IterateAndSynthesizeAgent - Comprehensive Analysis
**Goal**: Verify agent scans codebase and saves comprehensive analysis

- [ ] **Test 3.1**: Run analysis on test repository
  ```bash
  python agents/iterate_and_synthesize.py "Analyze this project" --repo /path/to/test_repo
  ```
  - Expected:
    - Analysis completes without errors
    - File created at: `project_storage/test_repo/comprehensive_analysis.pkl`

- [ ] **Test 3.2**: Verify analysis is saved and can be loaded
  ```bash
  python -c "
  import pickle
  from shared.storage_manager import get_project_file_path
  path = get_project_file_path('/path/to/test_repo', 'comprehensive_analysis.pkl')
  with open(path, 'rb') as f:
      analysis = pickle.load(f)
  print(f'Loaded analysis: {len(analysis)} characters')
  print(analysis[:200])  # First 200 chars
  "
  ```
  - Expected: Analysis content is readable

- [ ] **Test 3.3**: Re-run analysis (update/overwrite)
  ```bash
  python agents/iterate_and_synthesize.py "Different question" --repo /path/to/test_repo
  ```
  - Expected: Analysis file is updated with new content

### Phase 4: QueryAgent - Answer Questions from Stored Analysis
**Goal**: Verify QueryAgent uses stored analysis to answer questions

- [ ] **Test 4.1**: Run QueryAgent with simple question
  ```bash
  python agents/query_agent.py /path/to/test_repo "What does this project do?"
  ```
  - Expected:
    - Agent loads stored analysis
    - Generates answer based on full codebase context
    - No errors

- [ ] **Test 4.2**: QueryAgent with specific technical question
  ```bash
  python agents/query_agent.py /path/to/test_repo "What are the main components?"
  ```
  - Expected: Answer reflects understanding of full codebase

- [ ] **Test 4.3**: QueryAgent error handling (no analysis yet)
  ```bash
  python agents/query_agent.py /path/to/new_repo "What is this?"
  ```
  - Expected: Clear error message asking to run analysis first

### Phase 5: Multi-Project Support
**Goal**: Verify system can handle multiple projects independently

- [ ] **Test 5.1**: Analyze two different projects
  ```bash
  python agents/iterate_and_synthesize.py "Analyze" --repo /project/a
  python agents/iterate_and_synthesize.py "Analyze" --repo /project/b
  ```
  - Expected:
    - Both create separate storage: `project_storage/a/` and `project_storage/b/`
    - No conflicts or file overwrites

- [ ] **Test 5.2**: Query different projects independently
  ```bash
  python agents/query_agent.py /project/a "What is project A?"
  python agents/query_agent.py /project/b "What is project B?"
  ```
  - Expected: Each returns analysis specific to its project

### Phase 6: Ollama Connectivity
**Goal**: Verify Ollama integration with remote endpoint

- [ ] **Test 6.1**: Test Ollama endpoint connectivity
  ```bash
  curl http://clutchbox.local:11434/api/tags
  ```
  - Expected: List of available models

- [ ] **Test 6.2**: Verify agent uses correct endpoint
  - Check logs during analysis for endpoint confirmation
  - Or add debug prints to verify URL being used

### Phase 7: End-to-End Workflow
**Goal**: Complete workflow test

**Scenario**: Analyze a real repository and ask multiple questions

1. Pick a test repository (e.g., a small open-source project)
2. Run comprehensive analysis:
   ```bash
   python agents/iterate_and_synthesize.py "Provide comprehensive analysis" --repo /path/to/repo
   ```
3. Ask questions using QueryAgent:
   ```bash
   python agents/query_agent.py /path/to/repo "What is the main purpose?"
   python agents/query_agent.py /path/to/repo "How does the authentication work?"
   python agents/query_agent.py /path/to/repo "What are the key dependencies?"
   ```
4. Verify:
   - [ ] All questions are answered from full codebase context
   - [ ] Answers are consistent across questions
   - [ ] No errors or crashes
   - [ ] Storage is properly organized in `project_storage/repo_name/`

## Success Criteria

- ✅ All 20+ test cases pass
- ✅ No files created in project root (all in `project_storage/`)
- ✅ Multiple projects can coexist without conflicts
- ✅ QueryAgent provides answers using full codebase context
- ✅ All configuration comes from `.env` file
- ✅ Ollama integration works with remote endpoint

## Known Issues / To Investigate

- [ ] OllamaEmbeddings may need endpoint configuration (for embeddings)
- [ ] Need to verify embeddings also use Clutch box Llama
- [ ] Performance baseline for comprehensive analysis on large repos
- [ ] Interactive mode for QueryAgent (planned for future release)

## Notes

- The comprehensive analysis is saved automatically by IterateAndSynthesizeAgent
- Storage is completely project-scoped - no cross-project contamination
- All LLM configuration comes from `.env` - config.yaml uses `${VAR:default}` syntax
- Tests should be run sequentially (later tests depend on earlier ones completing)
