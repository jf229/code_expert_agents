"""
LLM Provider Interface for RAG Agent Prototypes

Supports multiple LLM providers while keeping embeddings local with Ollama.
"""

import os
import openai
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any
from .wca_service import WCAService


class LLMProvider(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, model: str = "gpt-4", api_key: str = None, extra_headers: dict = None):
        self.model = model
        self.extra_headers = extra_headers or {}
        
        # Create client with privacy headers
        headers = {}
        headers.update(self.extra_headers)
        
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            default_headers=headers
        )
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            # Add privacy-aware parameters
            extra_params = {}
            
            # Add no-training parameter if supported and requested
            if self.extra_headers.get("OpenAI-Opt-Out") == "true":
                # OpenAI doesn't currently have a direct no-training parameter,
                # but the header indicates intent for future compliance
                pass
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 4000),
                **extra_params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")


class ClaudeProvider(LLMProvider):
    """Claude API provider."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Claude API."""
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 4000),
                "temperature": kwargs.get("temperature", 0.1),
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            return response.json()["content"][0]["text"]
        except Exception as e:
            raise Exception(f"Claude API error: {e}")


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, model: str = "gemini-pro", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini API."""
        try:
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": kwargs.get("temperature", 0.1),
                    "maxOutputTokens": kwargs.get("max_tokens", 4000)
                }
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")


class OllamaProvider(LLMProvider):
    """Ollama local provider."""
    
    def __init__(self, model: str = "granite3.2:8b"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.1)
                }
            }
            
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")


class WCAProvider(LLMProvider):
    """Watson Code Assistant provider."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("WCA_API_KEY")
        if not self.api_key:
            raise ValueError("WCA API key is required")
        self.wca_service = WCAService(self.api_key)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using WCA API."""
        try:
            # Use WCA's architectural analysis method for structured responses
            payload = {
                "message_payload": {
                    "messages": [{"content": prompt, "role": "USER"}],
                    "chat_session_id": str(__import__('uuid').uuid4())
                }
            }
            
            response = self.wca_service._call_wca_api(payload)
            
            # Extract the response text from WCA's response format
            if (response and 'response' in response and 
                'message' in response['response'] and 
                'content' in response['response']['message']):
                return response['response']['message']['content']
            
            # Fallback if the structure is unexpected
            return str(response)
                
        except Exception as e:
            raise Exception(f"WCA API error: {e}")


def get_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory function to get LLM provider instance."""
    
    providers = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider, 
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
        "wca": WCAProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unsupported provider: {provider_type}")
    
    # Extract privacy headers if provided
    extra_headers = kwargs.pop("extra_headers", None)
    
    # Add extra_headers parameter for external providers
    if provider_type in ["openai", "claude", "gemini"] and extra_headers:
        kwargs["extra_headers"] = extra_headers
    
    return providers[provider_type](**kwargs)