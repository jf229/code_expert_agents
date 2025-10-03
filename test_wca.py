# test_wca.py
import yaml
import json
from shared import WCAService

def main():
    print("--- WCA API Test Script ---")

    # 1. Load configuration
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        api_key = config.get("llm", {}).get("wca_api_key")
        if not api_key:
            print("Error: WCA API key not found in config.yaml")
            return
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return

    # 2. Instantiate the WCAService
    wca_service = WCAService(api_key=api_key)

    # 3. Define a simple question
    question = "How many feet are in a mile?"
    print(f"Sending question to WCA: '{question}'")

    # 4. Call a modified, simpler method in WCAService for basic chat
    try:
        # We need a method for simple chat, let's add it to WCAService
        response = wca_service._classify_question_type(question)

        # 5. Print the full response
        print("\n--- Full API Response ---")
        print(json.dumps(response, indent=2))

        # 6. Print the content of the answer
        print("\n--- Extracted Answer ---")
        if response and 'response' in response and 'message' in response['response']:
            print(response['response']['message']['content'])
        else:
            print("Could not extract a valid answer from the response.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
