Great repo—looked through it. It already has clean entry points and a flat layout (e.g., `top_k_retrieval.py`, `graph_based_retrieval.py`, `multi_representation.py`, `data_ingestion.py`, `shared_services.py`, `llm_providers.py`, `config.yaml`). That makes it easy to bolt on a privacy layer for your IP-only threat model. ([GitHub][1])

Below is a drop-in plan (with concrete code changes) to make the project “privacy-first” while still letting you use an external LLM for analysis.

# What we’ll add

1. **Local embeddings + vector store (no code leaves your network)**
2. **Sanitizer**: AST/lexer-aware identifier aliasing, optional literal scrubbing, path redaction
3. **Policy Gate**: allow/deny lists, per-query “context budget,” audit log of exactly what was sent
4. **Provider Hardening**: enforce “no retention / no training” flags, regional endpoints, and a default “local-only” mode
5. **De-aliasing** (optional) when applying suggested patches locally

# Touch points in this repo

* `data_ingestion.py`: build local index (vectors + metadata), optionally precompute call graphs
* `shared_services.py`: add Sanitizer + PolicyGate and a local retriever wrapper
* `top_k_retrieval.py`, `graph_based_retrieval.py`, `multi_representation.py`: swap in the local retriever and gate before calling the LLM
* `llm_providers.py`: enforce provider privacy flags and safe defaults
* `config.yaml`: new `privacy:` section + local vector DB config ([GitHub][1])

---

## 1) Config: add a privacy section

Append to `config.yaml`:

```yaml
privacy:
  enable: true
  pseudonymize_identifiers: true
  strip_comments: true
  redact_literals: ["urls", "emails", "long_strings"]  # or []
  hide_paths: true
  denylist_globs: ["**/infra/**", "**/scripts/migrations/**"]
  allowlist_globs: ["**/*.py", "**/*.ts", "**/*.java"]
  max_chars_per_query: 18000
  max_snippets_per_query: 6
  audit_log_path: ".privacy/audit.log"

index:
  vector_store: "qdrant"        # or "chroma"
  path: ".index"
  embedding_model: "bge-small"   # local model name
  embedding_dim: 768
```

(These leverage the repo’s existing simplicity around `config.yaml` + single-file prototypes. ([GitHub][1]))

---

## 2) Local embeddings & vector store

In `requirements.txt` add:

```
qdrant-client==1.9.0
sentence-transformers==3.0.1
tree_sitter==0.21.3
pathspec==0.12.1
```

Update `data_ingestion.py` to:

* walk the repo using `allowlist_globs`/`denylist_globs`
* chunk code (function/class aware if possible)
* compute **local** embeddings (e.g., `sentence-transformers` bge-small)
* upsert into Qdrant/Chroma with metadata `{fid, lang, path_hash}` (not the raw path if `hide_paths: true`)

**Key idea:** vectors and raw code never leave your box; only tiny, sanitized snippets do later.

---

## 3) Sanitizer + Policy Gate (in `shared_services.py`)

Add two classes:

**Sanitizer**

* Uses `tree_sitter` for languages you care about (Python/TS/Java).
* Replaces identifiers with stable aliases per file (`ClassA → C1`, `funcX → F2`, `var → V3`).
* Optional literal handling (`"https://…"` → `<URL_1>`, long strings → `<STR_2>`).
* Scrubs comments if configured.
* Returns `(sanitized_text, alias_map_digest)`; store the alias map locally under `.privacy/aliases/<fid>.json`.

**PolicyGate**

* Applies globs, caps `max_chars_per_query`, trims to `max_snippets_per_query`.
* Redacts file paths → use short IDs (`F1234`) unless `hide_paths: false`.
* Writes an **audit record** with: timestamp, question hash, snippet IDs, character counts, provider.

Sketch (abridged):

```python
class PolicyGate:
    def __init__(self, cfg, auditor): ...
    def filter_hits(self, hits):
        # enforce allow/deny, cap k and chars
        return trimmed_hits

class Sanitizer:
    def __init__(self, cfg): ...
    def run(self, code, lang, fid):
        # tree-sitter parse, replace identifiers/comments/literals
        return sanitized, alias_digest
```

Expose a helper:

```python
def prepare_context(question, hits, cfg):
    hits = PolicyGate(cfg).filter_hits(hits)
    out = []
    for h in hits:
        s_text, digest = Sanitizer(cfg).run(h.text, h.lang, h.fid)
        out.append({"fid": h.fid, "text": s_text, "digest": digest})
    return out
```

---

## 4) Wire the retriever into the prototypes

In `top_k_retrieval.py`, `graph_based_retrieval.py`, and `multi_representation.py`, replace any direct provider-side embedding calls with:

```python
from shared_services import local_retriever, prepare_context, llm_answer

hits = local_retriever.search(question, top_k=cfg.retrieval.k)  # local vectors
ctx  = prepare_context(question, hits, cfg)                     # sanitize + gate
answer = llm_answer(question, ctx, cfg)                         # external LLM
```

These files are already meant to be thin “strategies,” so the swap is minimal. ([GitHub][1])

---

## 5) Provider hardening (`llm_providers.py`)

* Default provider to **local** (`ollama`) unless `privacy.enable=false`.
* For external providers, attach flags and safe defaults:

  * Disable data retention / training (per vendor option/parameter if exposed).
  * Prefer regional endpoints (set in env or config).
  * Add per-request header/comment reminding: “no training / no logging of inputs beyond 30 days” (and reflect in your DPA).

Also, take the sanitized `ctx` and build a compact prompt that **does not** include file paths or repo names when `hide_paths: true`.

---

## 6) Optional de-aliasing (local only)

If an agent proposes a patch, run a **local** de-alias pass (inverse of the alias map) before showing a diff to devs. Keep the alias maps in `.privacy/aliases/` keyed by `fid`.

---

## 7) Auditing & tests

* Write every outbound payload to `.privacy/audit.log` (hash the content; don’t store the full snippet unless required internally).
* Unit tests:

  * “No raw identifiers leave” when `pseudonymize_identifiers: true`.
  * Enforce char/snippet budgets.
  * Denylist respected.
  * Paths hidden when configured.

---

## 8) Example minimal diff (illustrative)

**`shared_services.py` (new bits only)**

```python
def llm_answer(question, ctx, cfg):
    # ctx = [{"fid": "F123", "text": "...", "digest": "a1b2"}]
    joined = "\n\n".join(f"[{c['fid']}]\\n{c['text']}" for c in ctx)
    prompt = f"You are a code expert. Answer the question using only the provided snippets.\n\n{joined}\n\nQ: {question}\nA:"
    return providers.call_llm(prompt, cfg)

class LocalRetriever:
    def __init__(self, cfg): ...
    def search(self, question, top_k=8):
        # embed question locally, search local vector store, return hits
        return hits

local_retriever = LocalRetriever(cfg)
```

**`llm_providers.py`**

```python
def call_llm(prompt, cfg):
    if cfg.llm.provider == "ollama":
        return _ollama_call(prompt, model=cfg.llm.model)
    if cfg.llm.provider in ("openai","claude","gemini","wca"):
        assert cfg.privacy.enable  # ensure we only send sanitized prompts
        return _external_call_with_privacy(prompt, cfg)

def _external_call_with_privacy(prompt, cfg):
    # set donottrain/retention flags if vendor supports them
    # choose regional endpoint
    # attach org/project IDs, not repo path
    return vendor_client.invoke(prompt, **privacy_kwargs(cfg))
```

---

## 9) Operational knobs

* **Fast mode**: `pseudonymize_identifiers=false`, only path hiding + denylist.
* **Strict mode**: full aliasing, comments stripped, literals redacted, small `max_chars_per_query`.
* **Air-gapped**: set provider to `ollama` only; you still get local RAG and summaries.

---

## Why this fits your threat model

* You said there are **no secrets/PII**; main risk is **IP leakage** via names, structure, and raw code.
* Identifier aliasing + path hiding preserve reasoning quality while protecting naming and architecture signals.
* Local embeddings + retrieval mean the repo itself is never shipped to a vendor; only **minimal, sanitized** spans go out.

If you want, I can turn this into a ready PR against that repo (new `privacy/` module, config updates, and the wiring changes above) and include a tiny demo repo so you can validate that usefulness isn’t lost with aliasing.

[1]: https://github.com/jf229/code_expert_agents "GitHub - jf229/code_expert_agents"

