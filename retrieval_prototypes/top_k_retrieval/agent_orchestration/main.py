# agent_orchestration/main.py
import os
import yaml
import pickle
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from wca_service import WCAService
from dotenv import load_dotenv

def main():
    print("Agent orchestration component (Top-K Retrieval Strategy)")
    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    llm_config = config["llm"]
    embedding_config = config["embeddings"]
    storage_config = config["storage"]
    retrieval_config = config["retrieval"]
    top_k = retrieval_config.get("top_k", 5)
    provider = llm_config.get("provider")

    retriever = get_retriever(embedding_config, storage_config, k=top_k)

    if provider == "wca":
        api_key = os.environ.get("WCA_API_KEY")
        if not api_key:
            raise ValueError("WCA_API_KEY not found in environment variables.")
        wca_service = WCAService(api_key=api_key)
        
        def wca_qa(inputs):
            question = inputs['question']
            retrieved_docs = retriever.get_relevant_documents(question)
            print(f"Retrieved {len(retrieved_docs)} documents for analysis.")
            try:
                response = wca_service.get_architectural_analysis(question, retrieved_docs)
                answer = response['response']['message']['content']
            except Exception as e:
                print(f"Error getting analysis from WCA: {e}")
                answer = "Sorry, I was unable to generate an analysis from WCA."
            return { "question": question, "answer": answer, "source_documents": retrieved_docs }
        
        print(f"Agent orchestration component initialized with WCA (retrieving top {top_k} documents).")
        return wca_qa

    else: # Default to Ollama
        llm = ChatOllama(
            model=llm_config["ollama_model"],
            temperature=0,
            top_k=40,
            top_p=0.9
        )
        
        # This is the corrected, working template
        _template = """You are a senior software architect. Based on the following set of retrieved source files, produce a comprehensive analysis of the project that directly answers the user's question.

Retrieved Documents (Context):
{context}

User's Question:
{question}

Architectural Analysis:"""
        PROMPT = PromptTemplate(template=_template, input_variables=["context", "question"])

        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        
        combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

        qa = ConversationalRetrievalChain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain,
            question_generator=LLMChain(llm=llm, prompt=PromptTemplate(template="{question}", input_variables=["question"])),
            return_source_documents=True,
        )
        
        print(f"Agent orchestration component initialized with Ollama (retrieving top {top_k} documents).")
        return qa

def get_retriever(embedding_config, storage_config, k):
    embedding = OllamaEmbeddings(model=embedding_config["model"])
    vectorstore = Chroma(collection_name="split_parents", persist_directory=storage_config["vector_store"], embedding_function=embedding)
    with open(storage_config["doc_store"], "rb") as f:
        store = pickle.load(f)
    retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=RecursiveCharacterTextSplitter(chunk_size=400), child_search_kwargs={'k': k})
    return retriever

if __name__ == "__main__":
    main()
