#!/usr/bin/env python3
"""
Iterate and Synthesize RAG Agent

This "MapReduce" agent analyzes every single file, making it extremely thorough 
but very slow. It processes all files in the repository and synthesizes them 
into a comprehensive analysis.

Usage:
    python iterate_and_synthesize.py "Provide a comprehensive architectural analysis of this entire project"

Use Case: Generating a complete, repository-wide architectural overview.
"""

import argparse
import os
import pickle
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain imports
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import shared services and central data ingestion
from shared_services import WCAService, ResponseGenerator, pull_ollama_model, load_config
from data_ingestion import main as data_ingestion_main


class IterateAndSynthesizeAgent:
    """Main Iterate and Synthesize Agent."""
    
    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self.llm_config = self.config["llm"]
        self.storage_config = self.config["storage"]
        self.provider = self.llm_config.get("provider")
        self.response_generator = ResponseGenerator(prototype_name="Iterate & Synthesize")
    
    def _setup_provider(self):
        """Setup LLM provider (WCA or Ollama)."""
        if self.provider == "wca":
            api_key = os.environ.get("WCA_API_KEY")
            if not api_key:
                raise ValueError("WCA_API_KEY not found in environment variables.")
            return WCAService(api_key=api_key)
        else:
            return ChatOllama(
                model=self.llm_config["ollama_model"],
                temperature=0,
                top_k=40,
                top_p=0.9
            )
    
    def _create_file_summary_chain(self, llm):
        """Create a chain for summarizing individual files."""
        prompt_template = """Summarize the purpose and key components of the following file. Focus on:
- What the file does
- Key classes, functions, or components
- How it fits into the overall architecture

File content:
{file_content}

Summary:"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["file_content"])
        return LLMChain(llm=llm, prompt=prompt)
    
    def _create_synthesis_chain(self, llm):
        """Create a chain for synthesizing all summaries."""
        prompt_template = """You are a senior software architect. Based on the following summaries of all the files in a repository, produce a comprehensive analysis of the project. Your analysis should be structured with the following sections:

1. **High-Level Summary:** A brief, one-paragraph overview of the project's purpose and primary function.
2. **Core Components:** A bulleted list of the key components, classes, or modules, with a brief description of each one's responsibility.
3. **Architectural Patterns:** An analysis of the architectural patterns used (e.g., MVVM, Singleton, Service-Oriented).
4. **Primary Use Cases:** A description of the main user stories or workflows that the application enables.
5. **Code Flow and Data Management:** An explanation of how data flows through the system, from the UI to the services and the database.

Here are the summaries:
{summaries}

Architectural Analysis:"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["summaries"])
        return LLMChain(llm=llm, prompt=prompt)
    
    def _process_all_files_wca(self, documents, wca_service):
        """Process all files using WCA service."""
        print("Summarizing all files using WCA...")
        
        summaries = []
        for doc in tqdm(documents, desc="Processing files"):
            try:
                file_path = doc.metadata.get('source', 'unknown')
                file_content = doc.page_content
                
                # Get summary from WCA
                response = wca_service.summarize_file(file_path, file_content)
                summary_content = response.get("choices", [{}])[0].get("message", {}).get("content", "No summary available")
                
                summary_text = f"**{os.path.basename(file_path)}:**\n{summary_content}\n"
                summaries.append(summary_text)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Combine all summaries
        all_summaries = "\n".join(summaries)
        
        # Get final architectural analysis
        print("Synthesizing architectural analysis using WCA...")
        try:
            response = wca_service.synthesize_summaries(all_summaries)
            final_analysis = response.get("choices", [{}])[0].get("message", {}).get("content", "No analysis available")
            return final_analysis
        except Exception as e:
            print(f"Error synthesizing summaries: {e}")
            return "Error generating final analysis."
    
    def _process_all_files_ollama(self, documents, llm):
        """Process all files using Ollama LLM."""
        print("Summarizing all files using Ollama...")
        
        # Create chains
        summary_chain = self._create_file_summary_chain(llm)
        synthesis_chain = self._create_synthesis_chain(llm)
        
        summaries = []
        for doc in tqdm(documents, desc="Processing files"):
            try:
                file_path = doc.metadata.get('source', 'unknown')
                file_content = doc.page_content
                
                # Get summary
                summary_response = summary_chain.run(file_content=file_content)
                summary_text = f"**{os.path.basename(file_path)}:**\n{summary_response}\n"
                summaries.append(summary_text)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Combine all summaries
        all_summaries = "\n".join(summaries)
        
        # Get final architectural analysis
        print("Synthesizing architectural analysis using Ollama...")
        try:
            final_analysis = synthesis_chain.run(summaries=all_summaries)
            return final_analysis
        except Exception as e:
            print(f"Error synthesizing summaries: {e}")
            return "Error generating final analysis."
    
    def run(self, question):
        """Run the Iterate and Synthesize agent with the given question."""
        print("--- RAG-based Code Expert Agent (Iterate and Synthesize) ---")
        
        # Step 1: Data Ingestion
        print("--- Data Ingestion ---")
        data_ingestion_main()
        
        # Step 2: Load all documents
        print("--- Loading Documents ---")
        raw_docs_path = self.storage_config["raw_docs"]
        with open(raw_docs_path, "rb") as f:
            documents = pickle.load(f)
        
        print(f"Loaded {len(documents)} documents for analysis.")
        
        # Step 3: Process all files and synthesize
        print("--- File Analysis and Synthesis ---")
        
        if self.provider == "wca":
            wca_service = self._setup_provider()
            final_analysis = self._process_all_files_wca(documents, wca_service)
        else:
            llm = self._setup_provider()
            final_analysis = self._process_all_files_ollama(documents, llm)
        
        # Step 4: Generate response
        print("--- Response Generation ---")
        qa_result = {
            "question": question,
            "answer": final_analysis,
            "source_documents": documents  # All documents are source documents
        }
        
        self.response_generator.generate_response(qa_result)
        
        print("--- System Finished ---")


def main():
    """Main entry point for Iterate and Synthesize prototype."""
    parser = argparse.ArgumentParser(description="RAG-based Code Expert Agent (Iterate and Synthesize)")
    parser.add_argument("question", type=str, help="The question to ask the agent.")
    args = parser.parse_args()
    
    # Load configuration and pull models if needed
    config = load_config()
    llm_model = config["llm"]["ollama_model"]
    embedding_model = config["embeddings"]["model"]
    
    pull_ollama_model(llm_model)
    pull_ollama_model(embedding_model)
    
    # Run the agent
    agent = IterateAndSynthesizeAgent()
    agent.run(args.question)


if __name__ == "__main__":
    main()