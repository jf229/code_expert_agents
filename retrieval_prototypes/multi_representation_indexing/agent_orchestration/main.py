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
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from wca_service import WCAService
from dotenv import load_dotenv

def main(strategy='specific'):
    print(f"Agent orchestration component (Multi-Representation Hybrid Strategy, Strategy: {strategy})")
    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    llm_config = config["llm"]
    embedding_config = config["embeddings"]
    storage_config = config["storage"]
    retrieval_config = config.get("retrieval", {})
    top_k = retrieval_config.get("top_k", 5)
    provider = llm_config.get("provider")

    if provider == "wca":
        api_key = os.environ.get("WCA_API_KEY")
        if not api_key: raise ValueError("WCA_API_KEY not found.")
        wca_service = WCAService(api_key=api_key)
    else:
        ollama_llm = ChatOllama(
            model=llm_config.get("ollama_model"),
            temperature=0,
            top_k=40,
            top_p=0.9
        )

    if strategy == 'broad':
        def qa_flow(inputs):
            # ... (rest of broad strategy is the same)
            question = inputs['question']
            print("Using 'Summarize the Summaries' strategy.")
            representations_path = storage_config["multi_representations"]
            try:
                with open(representations_path, "rb") as f: representations = pickle.load(f)
                summaries = [rep['summary'] for rep in representations]
                all_summaries = "\n\n---\n\n".join(summaries)
                if provider == "wca":
                    response = wca_service.synthesize_summaries(all_summaries)
                    answer = response['response']['message']['content']
                else: # Ollama
                    prompt = PromptTemplate(template="""You are a senior software architect...""", input_variables=["summaries"])
                    chain = LLMChain(llm=ollama_llm, prompt=prompt)
                    answer = chain.run(summaries=all_summaries)
                return { "question": question, "answer": answer, "source_documents": [] }
            except FileNotFoundError:
                return { "question": question, "answer": f"Could not find summaries file at {representations_path}. Please run representation_builder.py.", "source_documents": [] }
            except Exception as e:
                return { "question": question, "answer": f"An error occurred: {e}", "source_documents": [] }
        print("Agent orchestration component initialized for broad queries.")
        return qa_flow

    else: # Specific query strategy
        vectorstore = Chroma(collection_name="multi_representation", persist_directory=storage_config["vector_store"], embedding_function=OllamaEmbeddings(model=embedding_config["model"]))
        with open(storage_config["doc_store"], "rb") as f: store = pickle.load(f)
        retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id", search_kwargs={'k': top_k})
        
        if provider == "wca":
            def qa_flow(inputs):
                question = inputs['question']
                retrieved_docs = retriever.get_relevant_documents(question)
                print(f"Retrieved {len(retrieved_docs)} documents.")
                try:
                    response = wca_service.get_architectural_analysis(question, retrieved_docs)
                    answer = response['response']['message']['content']
                except Exception as e:
                    print(f"Error getting analysis from WA: {e}")
                    answer = "Sorry, I was unable to generate an analysis from WCA."
                return { "question": question, "answer": answer, "source_documents": retrieved_docs }
            print(f"Agent orchestration component initialized with WCA for specific queries.")
            return qa_flow
        else: # Ollama
            _template = """You are a senior software architect. Based on the following set of retrieved source files, produce a comprehensive analysis of the project that directly answers the user's question.

Retrieved Documents (Context):
{context}

User's Question:
{question}

Architectural Analysis:"""
            PROMPT = PromptTemplate(template=_template, input_variables=["context", "question"])
            llm_chain = LLMChain(llm=ollama_llm, prompt=PROMPT)
            question_generator = LLMChain(llm=ollama_llm, prompt=PromptTemplate(template="{question}", input_variables=["question"]))
            combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
            qa = ConversationalRetrievalChain(
                retriever=retriever,
                combine_docs_chain=combine_docs_chain,
                question_generator=question_generator,
                return_source_documents=True,
            )
            print(f"Agent orchestration component initialized with Ollama for specific queries.")
            return qa

if __name__ == "__main__":
    main()