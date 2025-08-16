# wca_service.py
import os
import requests
import time
import base64
import json
import uuid

class WCAService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.bearer_token = None
        self.token_expiry_time = 0
        self.iam_url = "https://iam.cloud.ibm.com/identity/token"
        self.wca_url = "https://api.dataplatform.cloud.ibm.com/v2/wca/core/chat/text/generation"

    def get_bearer_token(self):
        # ... (same as before)
        if self.bearer_token and time.time() < self.token_expiry_time:
            return self.bearer_token
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={self.api_key}"
        try:
            response = requests.post(self.iam_url, headers=headers, data=data)
            response.raise_for_status()
            token_data = response.json()
            self.bearer_token = token_data["access_token"]
            self.token_expiry_time = time.time() + token_data["expires_in"] - 60
            return self.bearer_token
        except requests.exceptions.RequestException as e:
            print(f"Error getting bearer token: {e}")
            raise

    def _call_wca_api(self, payload_dict, files_dict=None):
        # ... (same as before)
        try:
            token = self.get_bearer_token()
            headers = { "Authorization": f"Bearer {token}", "Request-ID": str(uuid.uuid4()) }
            payload_b64_string = base64.b64encode(json.dumps(payload_dict).encode('utf-8')).decode('utf-8')
            files_to_send = [('message', (None, payload_b64_string))]
            if files_dict:
                for file_path, file_content in files_dict.items():
                    encoded_content = base64.b64encode(file_content.encode()).decode('utf-8')
                    files_to_send.append(('files', (os.path.basename(file_path), encoded_content, 'text/plain')))
            response = requests.post(self.wca_url, headers=headers, files=files_to_send, timeout=180)
            if not response.ok:
                print(f"Error from WCA API. Status: {response.status_code}. Response: {response.text}")
                response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling WCA API: {e}")
            raise

    def summarize_file(self, file_path, file_content):
        # ... (same as before)
        prompt = f"Provide a concise, one-paragraph summary of the following file's purpose and key components: [{os.path.basename(file_path)}](<file-{os.path.basename(file_path)}>)"
        payload = { "message_payload": { "messages": [{"content": prompt, "role": "USER"}], "chat_session_id": str(uuid.uuid4()) } }
        files_data = {file_path: file_content}
        return self._call_wca_api(payload, files_data)

    def generate_hypothetical_questions(self, file_path, file_content):
        # ... (same as before)
        prompt = f"Based on the content of the file [{os.path.basename(file_path)}](<file-{os.path.basename(file_path)}>), generate a list of 3-5 high-level questions that this file could help answer. Return only the questions, each on a new line."
        payload = { "message_payload": { "messages": [{"content": prompt, "role": "USER"}], "chat_session_id": str(uuid.uuid4()) } }
        files_data = {file_path: file_content}
        return self._call_wca_api(payload, files_data)

    def synthesize_summaries(self, summaries):
        """Asks WCA for a detailed architectural analysis based on summaries."""
        prompt = """You are a senior software architect. Based on the following summaries of all the files in a repository, produce a comprehensive analysis of the project. Your analysis should be structured with the following sections:

1.  **High-Level Summary:** A brief, one-paragraph overview of the project's purpose and primary function.
2.  **Core Components:** A bulleted list of the key components, classes, or modules, with a brief description of each one's responsibility.
3.  **Architectural Patterns:** An analysis of the architectural patterns used (e.g., MVVM, Singleton, Service-Oriented).
4.  **Primary Use Cases:** A description of the main user stories or workflows that the application enables.
5.  **Code Flow and Data Management:** An explanation of how data flows through the system, from the UI to the services and the database.

Here are the summaries:
""" + summaries
        payload = {
            "message_payload": {
                "messages": [{"content": prompt, "role": "USER"}],
                "chat_session_id": str(uuid.uuid4())
            }
        }
        return self._call_wca_api(payload)

    def get_architectural_analysis(self, question, documents):
        """Asks WCA for a detailed architectural analysis based on a set of documents."""
        file_references = " ".join([f"[{os.path.basename(doc.metadata['source'])}](<file-{os.path.basename(doc.metadata['source'])}>)" for doc in documents])
        
        prompt = f"""You are a senior software architect. The user has asked the following question: '{question}'

Based on the following set of retrieved source files, produce a comprehensive analysis of the project. Your analysis should be structured with the following sections:

1.  **High-Level Summary:** A brief, one-paragraph overview of the project's purpose and primary function, answering the user's question directly.
2.  **Core Components:** A bulleted list of the key components, classes, or modules present in the provided files, with a brief description of each one's responsibility.
3.  **Architectural Patterns:** An analysis of the architectural patterns suggested by the provided files (e.g., MVVM, Singleton, Service-Oriented).
4.  **Code Flow and Data Management:** An explanation of how data likely flows through the system, based on the interactions between the components in the provided files.

Retrieved Files: {file_references}
"""
        payload = {
            "message_payload": {
                "messages": [{"content": prompt, "role": "USER"}],
                "chat_session_id": str(uuid.uuid4())
            }
        }
        files_data = {doc.metadata['source']: doc.page_content for doc in documents}
        return self._call_wca_api(payload, files_data)
