# agent_orchestration/main.py
import os
import yaml
import pickle
import networkx as nx
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from wca_service import WCAService
from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document

def load_graph(graph_path):
    if not os.path.exists(graph_path): return None
    with open(graph_path, "rb") as f: return pickle.load(f)

def retrieve_from_graph(graph, query):
    relevant_files = set()
    query_terms = query.lower().split()
    for node, data in graph.nodes(data=True):
        node_name = data.get('name', '').lower()
        node_id = str(node).lower()
        if any(term in node_name for term in query_terms) or any(term in node_id for term in query_terms):
            if data.get('type') == 'file': relevant_files.add(node)
            elif 'file' in data: relevant_files.add(data['file'])
    return list(relevant_files)

def main():
    print("Agent orchestration component (Graph-Based Retrieval Strategy)")
    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    llm_config = config["llm"]
    storage_config = config["storage"]
    provider = llm_config.get("provider")

    graph_path = storage_config["code_graph"]
    graph = load_graph(graph_path)
    if graph is None:
        print(f"Error: code_graph.gpickle not found at {graph_path}")
        return None

    class GraphRetriever(BaseRetriever):
        """A custom retriever that fetches document content based on a graph search."""
        def _get_relevant_documents(self, query: str):
            relevant_files = retrieve_from_graph(graph, query)
            retrieved_docs = []
            for file_path in relevant_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    retrieved_docs.append(Document(page_content=content, metadata={'source': file_path}))
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")
            return retrieved_docs
        
        async def _aget_relevant_documents(self, query: str):
            return self._get_relevant_documents(query)

    retriever = GraphRetriever()

    if provider == "wca":
        api_key = os.environ.get("WCA_API_KEY")
        if not api_key: raise ValueError("WCA_API_KEY not found.")
        wca_service = WCAService(api_key=api_key)
        def qa_flow(inputs):
            question = inputs['question']
            retrieved_docs = retriever.get_relevant_documents(question)
            print(f"Retrieved {len(retrieved_docs)} relevant files from the graph.")
            if not retrieved_docs: return { "question": question, "answer": "Sorry, I could not find any relevant files...", "source_documents": [] }
            try:
                response = wca_service.get_architectural_analysis(question, retrieved_docs)
                answer = response['response']['message']['content']
            except Exception as e:
                print(f"Error getting analysis from WCA: {e}")
                answer = "Sorry, I was unable to generate an analysis from WCA."
            return { "question": question, "answer": answer, "source_documents": retrieved_docs }
        print("Agent orchestration component initialized with WCA.")
        return qa_flow

    else: # Default to Ollama
        llm = ChatOllama(
            model=llm_config["ollama_model"],
            temperature=0,
            top_k=40,
            top_p=0.9
        )
        
        _template = """You are a senior software architect. Based on the following set of retrieved source files, produce a comprehensive analysis of the project that directly answers the user's question.

Retrieved Documents (Context):
{context}

User's Question:
{question}

Architectural Analysis:"""
        PROMPT = PromptTemplate(template=_template, input_variables=["context", "question"])

        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        question_generator = LLMChain(llm=llm, prompt=PromptTemplate(template="{question}", input_variables=["question"]))
        combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
            
        qa = ConversationalRetrievalChain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain,
            question_generator=question_generator,
            return_source_documents=True,
        )
        
        print("Agent orchestration component initialized with Ollama.")
        return qa

if __name__ == "__main__":
    main()