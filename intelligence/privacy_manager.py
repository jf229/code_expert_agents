"""
Privacy Manager for RAG Code Expert Agents

Provides basic sanitization and policy enforcement for protecting IP 
when sending code snippets to external LLMs.
"""

import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class PrivacyManager:
    """Handles code sanitization and privacy policy enforcement."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.privacy_config = config.get("privacy", {})
        self.enabled = self.privacy_config.get("enable", False)
        self.mode = self.privacy_config.get("mode", "fast")
        
        # Create privacy directories
        self._ensure_privacy_dirs()
        
        # Load or create alias mappings
        self.file_aliases = {}
        self.identifier_aliases = {}
        
    def _ensure_privacy_dirs(self):
        """Create necessary privacy directories."""
        privacy_dir = Path(".privacy")
        privacy_dir.mkdir(exist_ok=True)
        
        (privacy_dir / "aliases").mkdir(exist_ok=True)
        (privacy_dir / "audit").mkdir(exist_ok=True)
    
    def sanitize_documents(self, documents: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Sanitize a list of documents for external LLM consumption.
        
        Returns:
            Tuple of (sanitized_documents, metadata_for_audit)
        """
        if not self.enabled:
            return documents, {"privacy_enabled": False}
        
        sanitized_docs = []
        audit_metadata = {
            "privacy_enabled": True,
            "mode": self.mode,
            "original_count": len(documents),
            "sanitized_count": 0,
            "total_chars": 0,
            "files_processed": []
        }
        
        # Apply query limits
        max_snippets = self.privacy_config.get("max_snippets_per_query", 6)
        max_chars = self.privacy_config.get("max_chars_per_query", 18000)
        
        char_count = 0
        processed_count = 0
        
        for doc in documents[:max_snippets]:
            if char_count >= max_chars:
                break
                
            sanitized_doc, file_metadata = self._sanitize_single_document(doc)
            
            # Check if adding this doc would exceed char limit
            new_char_count = char_count + len(sanitized_doc.page_content)
            if new_char_count > max_chars and processed_count > 0:
                break
                
            sanitized_docs.append(sanitized_doc)
            char_count = new_char_count
            processed_count += 1
            
            audit_metadata["files_processed"].append(file_metadata)
        
        audit_metadata["sanitized_count"] = processed_count
        audit_metadata["total_chars"] = char_count
        
        return sanitized_docs, audit_metadata
    
    def _sanitize_single_document(self, doc: Any) -> Tuple[Any, Dict[str, Any]]:
        """Sanitize a single document."""
        source_path = doc.metadata.get("source", "unknown")
        content = doc.page_content
        
        # Generate file alias
        file_id = self._get_file_alias(source_path)
        
        # Sanitize content
        sanitized_content = content
        
        if self.privacy_config.get("pseudonymize_identifiers", True):
            sanitized_content = self._pseudonymize_identifiers(sanitized_content, source_path)
        
        if self.privacy_config.get("strip_comments", False):
            sanitized_content = self._strip_comments(sanitized_content)
        
        if self.privacy_config.get("redact_literals", []):
            sanitized_content = self._redact_literals(sanitized_content)
        
        # Create sanitized document
        sanitized_doc = type(doc)(
            page_content=sanitized_content,
            metadata={
                "source": file_id if self.privacy_config.get("hide_paths", True) else source_path,
                "file_id": file_id,
                "original_source": source_path if self.privacy_config.get("hide_paths", True) else None
            }
        )
        
        file_metadata = {
            "file_id": file_id,
            "original_path": source_path,
            "original_chars": len(content),
            "sanitized_chars": len(sanitized_content),
            "reduction_ratio": 1.0 - (len(sanitized_content) / len(content)) if len(content) > 0 else 0
        }
        
        return sanitized_doc, file_metadata
    
    def _get_file_alias(self, file_path: str) -> str:
        """Generate a consistent alias for a file path."""
        if file_path in self.file_aliases:
            return self.file_aliases[file_path]
        
        # Create a short hash-based ID
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        file_id = f"F{len(self.file_aliases) + 1:03d}_{file_hash}"
        
        self.file_aliases[file_path] = file_id
        return file_id
    
    def _pseudonymize_identifiers(self, content: str, file_path: str) -> str:
        """Replace identifiers with aliases using simple regex patterns."""
        if self.mode == "fast":
            return self._fast_pseudonymize(content, file_path)
        else:
            # For now, use fast mode. Strict mode would use tree-sitter
            return self._fast_pseudonymize(content, file_path)
    
    def _fast_pseudonymize(self, content: str, file_path: str) -> str:
        """Fast pseudonymization using regex patterns."""
        # Get or create identifier map for this file
        file_key = hashlib.md5(file_path.encode()).hexdigest()
        if file_key not in self.identifier_aliases:
            self.identifier_aliases[file_key] = {}
        
        id_map = self.identifier_aliases[file_key]
        
        # Pattern for class names (PascalCase)
        class_pattern = r'\b[A-Z][a-zA-Z0-9]*(?=\s*[\(:]|\s+extends|\s+implements)'
        content = self._replace_pattern(content, class_pattern, id_map, "C")
        
        # Pattern for function/method names (camelCase and snake_case)
        func_pattern = r'\b[a-z_][a-zA-Z0-9_]*(?=\s*\()'
        content = self._replace_pattern(content, func_pattern, id_map, "F")
        
        # Pattern for variable names (common patterns)
        var_pattern = r'\b[a-z_][a-zA-Z0-9_]*(?=\s*[=\+\-\*\/])'
        content = self._replace_pattern(content, var_pattern, id_map, "V")
        
        return content
    
    def _replace_pattern(self, content: str, pattern: str, id_map: Dict[str, str], prefix: str) -> str:
        """Replace matches of a pattern with aliases."""
        def replace_func(match):
            identifier = match.group(0)
            
            # Skip common keywords and short identifiers
            if identifier in {'if', 'for', 'while', 'def', 'class', 'import', 'from', 'return'} or len(identifier) <= 2:
                return identifier
            
            if identifier not in id_map:
                counter = len([k for k in id_map.keys() if k.startswith(prefix)]) + 1
                id_map[identifier] = f"{prefix}{counter}"
            
            return id_map[identifier]
        
        return re.sub(pattern, replace_func, content)
    
    def _strip_comments(self, content: str) -> str:
        """Remove comments from code."""
        # Remove Python-style comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        
        # Remove JavaScript/Java-style comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        return content
    
    def _redact_literals(self, content: str) -> str:
        """Redact literal values based on configuration."""
        redact_types = self.privacy_config.get("redact_literals", [])
        
        if "urls" in redact_types:
            content = re.sub(r'https?://[^\s\'"]+', '<URL>', content)
        
        if "emails" in redact_types:
            content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', content)
        
        if "long_strings" in redact_types:
            # Replace string literals longer than 50 characters
            content = re.sub(r'(["\'])([^"\']{50,})\1', r'\1<LONG_STRING>\1', content)
        
        return content
    
    def log_audit_event(self, event_type: str, metadata: Dict[str, Any], question: str = None):
        """Log an audit event for compliance tracking."""
        if not self.enabled:
            return
        
        audit_log_path = self.privacy_config.get("audit_log", ".privacy/audit.log")
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "privacy_mode": self.mode,
            "question_hash": hashlib.sha256(question.encode()).hexdigest()[:16] if question else None,
            "metadata": metadata
        }
        
        # Ensure directory exists
        Path(audit_log_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(audit_log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def get_alias_mappings(self) -> Dict[str, Any]:
        """Get current alias mappings for debugging/de-aliasing."""
        return {
            "file_aliases": self.file_aliases,
            "identifier_aliases": self.identifier_aliases
        }
    
    def save_alias_mappings(self):
        """Save alias mappings to disk for persistence."""
        alias_file = Path(".privacy/aliases/current_session.json")
        
        mappings = {
            "file_aliases": self.file_aliases,
            "identifier_aliases": self.identifier_aliases,
            "created_at": datetime.utcnow().isoformat()
        }
        
        with open(alias_file, "w") as f:
            json.dump(mappings, f, indent=2)


class PolicyGate:
    """Enforces privacy policies and additional restrictions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.privacy_config = config.get("privacy", {})
    
    def should_deny_file(self, file_path: str) -> bool:
        """Check if a file should be denied based on privacy policies."""
        import fnmatch
        
        denylist = self.privacy_config.get("denylist_globs", [])
        
        for pattern in denylist:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        
        return False
    
    def enforce_provider_safety(self, provider_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add safety parameters to provider configuration."""
        if not self.privacy_config.get("enable", False):
            return provider_config
        
        # Add no-training flags based on provider
        provider_type = provider_config.get("provider", "")
        
        if provider_type == "openai" and self.privacy_config.get("require_no_training", True):
            provider_config["extra_headers"] = provider_config.get("extra_headers", {})
            provider_config["extra_headers"]["OpenAI-Opt-Out"] = "true"
        
        # Add custom headers for compliance
        if "extra_headers" not in provider_config:
            provider_config["extra_headers"] = {}
        
        provider_config["extra_headers"]["X-Privacy-Mode"] = self.privacy_config.get("mode", "fast")
        
        return provider_config