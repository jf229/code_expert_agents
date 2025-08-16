# representation_builder.py
import os
import yaml
import pickle
from wca_service import WCAService
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tqdm import tqdm
import sys

def get_llm_chain(llm, prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["file_content"])
    return LLMChain(llm=llm, prompt=prompt)

def main():
    print("--- Multi-Representation Builder ---")

    # All paths are now relative to the project root
    config_path = "config.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Root configuration file not found at {config_path}")
        sys.exit(1)

    llm_config = config.get("llm", {})
    storage_config = config.get("storage", {})
    provider = llm_config.get("provider")

    if provider == "wca":
        # This needs to load from the root .env file
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("WCA_API_KEY")
        if not api_key:
            print("Error: WCA_API_KEY not found in environment. Please create a .env file in the project root.")
            sys.exit(1)
        wca_service = WCAService(api_key=api_key)
    elif provider == "ollama":
        ollama_llm = ChatOllama(model=llm_config.get("ollama_model"))
        summarize_chain = get_llm_chain(ollama_llm, "Provide a concise, one-paragraph summary of the following file's purpose and key components:\n\n{file_content}")
        questions_chain = get_llm_chain(ollama_llm, "Based on the content of the following file, generate a list of 3-5 high-level questions that this file could help answer. Return only the questions, each on a new line.\n\n{file_content}")
    else:
        print(f"Error: Provider '{provider}' is not supported.")
        sys.exit(1)

    raw_docs_path = storage_config.get("raw_docs")
    try:
        with open(raw_docs_path, "rb") as f:
            raw_docs = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: raw_documents.pkl not found at {raw_docs_path}")
        print("Please run the data_ingestion script in the top_k_retrieval prototype first.")
        sys.exit(1)

    representations = []
    print(f"Generating summaries and questions for {len(raw_docs)} documents using provider: {provider}...")

    for doc in tqdm(raw_docs, desc="Generating Representations"):
        file_path = doc.metadata['source']
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if provider == "wca":
                summary_response = wca_service.summarize_file(file_path, content)
                summary = summary_response['response']['message']['content']
                questions_response = wca_service.generate_hypothetical_questions(file_path, content)
                questions = questions_response['response']['message']['content'].split('\n')
            else: # Ollama
                summary = summarize_chain.run(file_content=content)
                questions = questions_chain.run(file_content=content).split('\n')

            representations.append({
                "source": file_path,
                "original_content": content,
                "summary": summary,
                "hypothetical_questions": questions
            })
        except Exception as e:
            print(f"\nCould not process {file_path}: {e}")

    output_path = storage_config.get("multi_representations")
    with open(output_path, "wb") as f:
        pickle.dump(representations, f)

    print(f"\nSuccessfully generated and saved representations for {len(representations)} documents.")
    print(f"Representations saved to: {output_path}")

if __name__ == "__main__":
    main()
