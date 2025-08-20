"""
Response generation classes for handling LLM outputs and formatting.
"""

import os
from .llm_providers import get_llm_provider
from .wca_service import WCAService


class UnifiedResponseGenerator:
    """Unified response generator that works with any LLM provider."""
    
    def __init__(self, config, show_sources=True, prototype_name="RAG Agent"):
        self.show_sources = show_sources
        self.prototype_name = prototype_name
        self.config = config
        
        # Initialize privacy manager
        from intelligence import PrivacyManager, PolicyGate
        self.privacy_manager = PrivacyManager(config)
        self.policy_gate = PolicyGate(config)
        
        # Setup LLM provider
        llm_config = config.get("llm", {})
        provider_type = llm_config.get("provider", "ollama")
        
        # Use unified provider interface with privacy hardening
        model = llm_config.get("models", {}).get(provider_type, "granite3.2:8b")
        
        # Apply privacy policies to provider config
        provider_config = {"provider": provider_type, "model": model}
        provider_config = self.policy_gate.enforce_provider_safety(provider_config)
        
        # Extract privacy headers for provider initialization
        extra_headers = provider_config.get("extra_headers")
        provider_kwargs = {"model": model} if provider_type != "wca" else {}
        if extra_headers:
            provider_kwargs["extra_headers"] = extra_headers
        
        self.llm_provider = get_llm_provider(provider_type, **provider_kwargs)
        
        # Keep WCA service for legacy methods that still need it
        if provider_type == "wca":
            api_key = os.environ.get("WCA_API_KEY")
            if not api_key:
                raise ValueError("WCA_API_KEY not found in environment variables.")
            self.wca_service = WCAService(api_key=api_key)
        else:
            self.wca_service = None
    
    def generate_response_with_context(self, question, documents):
        """Generate response using LLM provider with retrieved documents."""
        
        # Apply privacy sanitization if enabled
        sanitized_documents, audit_metadata = self.privacy_manager.sanitize_documents(documents)
        
        # Log privacy audit event
        if self.privacy_manager.enabled:
            self.privacy_manager.log_audit_event("query_processed", audit_metadata, question)
            print(f"Privacy: Sanitized {audit_metadata['sanitized_count']}/{audit_metadata['original_count']} documents ({audit_metadata['total_chars']} chars)")
        
        # Prepare context from sanitized documents
        context = self._format_context(sanitized_documents)
        
        # Create prompt
        prompt = f"""You are a senior software engineer analyzing a codebase. Based on the retrieved code context below, provide a comprehensive answer to the user's question.

Retrieved Code Context:
{context}

User's Question: {question}

Provide a detailed technical analysis:"""

        try:
            # Use unified provider interface for all providers including WCA
            answer = self.llm_provider.generate_response(
                prompt,
                temperature=self.config.get("llm", {}).get("temperature", 0.1),
                max_tokens=self.config.get("llm", {}).get("max_tokens", 4000)
            )
            
            # Log successful external LLM call
            if self.privacy_manager.enabled and self.config.get("llm", {}).get("provider") != "ollama":
                self.privacy_manager.log_audit_event("external_llm_call", {
                    "provider": self.config.get("llm", {}).get("provider"),
                    "prompt_chars": len(prompt),
                    "response_chars": len(answer) if answer else 0
                })
            
            # Format and display result (using original documents for source display)
            qa_result = {
                "question": question,
                "answer": answer,
                "source_documents": documents,  # Show original sources to user
                "privacy_metadata": audit_metadata if self.privacy_manager.enabled else None
            }
            
            self.generate_response(qa_result)
            return qa_result
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Log error event
            if self.privacy_manager.enabled:
                self.privacy_manager.log_audit_event("error", {"error": str(e), "question_hash": question[:50]})
            return None
    
    def _format_context(self, documents):
        """Format documents into context string."""
        context_parts = []
        
        for i, doc in enumerate(documents[:10]):  # Limit to top 10 documents
            source = doc.metadata.get('source', f'Document {i+1}')
            content = doc.page_content[:2000]  # Limit content length
            context_parts.append(f"=== {source} ===\n{content}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(self, qa_result):
        """Generate and display response from QA result."""
        print(f"\n--- {self.prototype_name} Response ---")
        print(f"Question: {qa_result['question']}")
        print(f"Answer: {qa_result['answer']}")
        
        if self.show_sources:
            source_documents = qa_result.get('source_documents', [])
            if source_documents:
                print("\nSource Documents:")
                for doc in source_documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        print(f"- {doc.metadata['source']}")


class ResponseGenerator:
    """Handles response generation and formatting for QA results."""
    
    def __init__(self, show_sources=True, prototype_name="RAG Agent"):
        self.show_sources = show_sources
        self.prototype_name = prototype_name
    
    def generate_response(self, qa_result):
        """Generate and display response from QA result."""
        print("Generating response...")
        print(f"Question: {qa_result['question']}")
        print(f"Answer: {qa_result['answer']}")
        
        if self.show_sources:
            source_documents = qa_result.get('source_documents', [])
            if source_documents:
                print("Source Documents:")
                for doc in source_documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        print(f"- {doc.metadata['source']}")
    
    def process_query(self, qa, question):
        """Process a query through the QA system and generate response."""
        print(f"Reasoning and response generation component ({self.prototype_name})")
        print("Sending query to the language model...")
        
        try:
            result = qa({"question": question, "chat_history": []})
            self.generate_response(result)
            return result
        except Exception as e:
            print(f"An error occurred during reasoning: {e}")
            return None