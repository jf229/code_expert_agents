# reasoning_response_generation/main.py
import sys

def generate_response(qa_result):
    print("Generating response...")
    print(f"Question: {qa_result['question']}")
    print(f"Answer: {qa_result['answer']}")
    source_documents = qa_result.get('source_documents', [])
    if source_documents:
        print("Source Documents:")
        for doc in source_documents:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                print(f"- {doc.metadata['source']}")

def main(qa, question):
    print("Reasoning and response generation component")
    print("Sending query to the language model...")
    
    try:
        result = qa({"question": question, "chat_history": []})
        generate_response(result)
    except Exception as e:
        print(f"An error occurred during reasoning: {e}")

if __name__ == "__main__":
    pass