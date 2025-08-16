# agent_orchestration/main.py
import os
import yaml
import pickle
from wca_service import WCAService
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tqdm import tqdm
from dotenv import load_dotenv

def main():
    print("Agent orchestration component (Iterate and Synthesize Strategy)")
    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    llm_config = config["llm"]
    storage_config = config["storage"]
    provider = llm_config.get("provider")

    if provider == "wca":
        api_key = os.environ.get("WCA_API_KEY")
        if not api_key:
            raise ValueError("WCA_API_KEY not found in environment variables.")
        wca_service = WCAService(api_key=api_key)
    else: # Ollama
        ollama_llm = ChatOllama(
            model=llm_config.get("ollama_model"),
            temperature=0,
            top_k=40,
            top_p=0.9
        )
        summarize_chain = LLMChain(llm=ollama_llm, prompt=PromptTemplate.from_template("Summarize..."))
        architectural_chain = LLMChain(llm=ollama_llm, prompt=PromptTemplate.from_template("You are a senior software architect..."))

    try:
        with open(storage_config["raw_docs"], "rb") as f:
            all_docs = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {storage_config['raw_docs']} not found.")
        return None

    def qa_flow(inputs):
        question = inputs['question']
        print(f"\n--- Summarizing {len(all_docs)} files (Map Step) using {provider} ---")
        summaries = []
        for doc in tqdm(all_docs, desc="Summarizing files"):
            file_path = doc.metadata['source']
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if provider == "wca":
                    response = wca_service.summarize_file(file_path, content)
                    summary = response['response']['message']['content']
                else: # Ollama
                    summary = summarize_chain.run(file_content=content)
                summaries.append(f"Summary for {os.path.basename(file_path)}:\n{summary}\n")
            except Exception as e:
                print(f"\nCould not summarize {file_path}: {e}")
        
        print("--- Finished Summarizing ---\n")
        print(f"--- Performing Architectural Analysis (Reduce Step) using {provider} ---")
        all_summaries = "\n---\n".join(summaries)
        
        try:
            if provider == "wca":
                response = wca_service.architectural_analysis(all_summaries)
                final_answer = response['response']['message']['content']
            else: # Ollama
                final_answer = architectural_chain.run(summaries=all_summaries)
        except Exception as e:
            print(f"Error synthesizing summaries: {e}")
            final_answer = "Sorry, I was unable to synthesize a final answer."
        return { "question": question, "answer": final_answer, "source_documents": all_docs }
    
    print("Agent orchestration component initialized.")
    return qa_flow

if __name__ == "__main__":
    main()
