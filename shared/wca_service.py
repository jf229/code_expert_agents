"""
Watson Code Assistant service for all RAG prototypes.
"""

import os
import requests
import time
import base64
import json
import uuid


class WCAService:
    """Watson Code Assistant service for all RAG prototypes."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.bearer_token = None
        self.token_expiry_time = 0
        self.iam_url = "https://iam.cloud.ibm.com/identity/token"
        self.wca_url = "https://api.dataplatform.cloud.ibm.com/v2/wca/core/chat/text/generation"

    def get_bearer_token(self):
        """Get or refresh the bearer token for WCA API access."""
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
        """Make a call to the WCA API with optional file attachments."""
        try:
            token = self.get_bearer_token()
            request_id = str(uuid.uuid4())
            headers = {
                "Authorization": f"Bearer {token}",
                "Request-ID": request_id
            }
            
            print(f"--- Sending request to WCA API ---")
            print(f"Request ID: {request_id}")
            print(f"URL: {self.wca_url}")
            
            payload_b64_string = base64.b64encode(
                json.dumps(payload_dict).encode('utf-8')
            ).decode('utf-8')
            
            files_to_send = [('message', (None, payload_b64_string))]
            
            if files_dict:
                print(f"Attaching {len(files_dict)} files to the request.")
                for file_path, file_content in files_dict.items():
                    encoded_content = base64.b64encode(
                        file_content.encode()
                    ).decode('utf-8')
                    files_to_send.append((
                        'files',
                        (os.path.basename(file_path), encoded_content, 'text/plain')
                    ))
            
            response = requests.post(
                self.wca_url,
                headers=headers,
                files=files_to_send,
                timeout=180
            )
            
            print(f"--- WCA API Response ---")
            print(f"Status Code: {response.status_code}")
            
            if not response.ok:
                print(f"Error from WCA API. Status: {response.status_code}. Response: {response.text}")
                response.raise_for_status()
            
            print("Request successful.")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling WCA API: {e}")
            raise

    def _classify_question_type(self, question):
        """Classify question to use appropriate prompt strategy."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['architecture', 'overview', 'structure', 'design', 'pattern']):
            return 'architectural'
        elif any(word in question_lower for word in ['how', 'implement', 'work', 'process', 'algorithm']):
            return 'implementation'
        elif any(word in question_lower for word in ['what', 'class', 'function', 'method', 'specific']):
            return 'entity_specific'
        elif any(word in question_lower for word in ['why', 'reason', 'purpose', 'decision', 'rationale']):
            return 'rationale'
        elif any(word in question_lower for word in ['flow', 'data', 'process', 'workflow', 'pipeline']):
            return 'data_flow'
        else:
            return 'general'

    def _get_prompt_for_question_type(self, question, question_type, file_references):
        """Return optimized prompt template for question type."""
        
        prompts = {
            'architectural': f"""You are a senior software architect analyzing system design. The user wants to understand the architectural aspects of this codebase.

User's Question: '{question}'

Focus your analysis on:
- Overall system design and architectural patterns
- Component relationships and dependencies  
- Design decisions and trade-offs
- System boundaries and interfaces
- Scalability and maintainability considerations

Structure your response with:
1. **Architectural Overview:** High-level system design answering the question directly
2. **Key Components:** Main architectural components and their responsibilities
3. **Design Patterns:** Architectural patterns and design principles used
4. **Component Relationships:** How components interact and depend on each other
5. **Design Rationale:** Why this architecture was chosen and its benefits

Retrieved Files: {file_references}""",

            'implementation': f"""You are a senior developer explaining code implementation. The user wants to understand HOW something works in this codebase.

User's Question: '{question}'

Focus your analysis on:
- Step-by-step process flows and algorithms
- Key implementation details and mechanisms
- Code execution paths and control flow
- Data transformations and processing logic

Structure your response with:
1. **Process Overview:** High-level explanation of how it works
2. **Implementation Steps:** Detailed step-by-step breakdown
3. **Key Algorithms:** Important algorithms and logic used
4. **Code Flow:** How execution flows through the components
5. **Technical Details:** Important implementation specifics

Retrieved Files: {file_references}""",

            'entity_specific': f"""You are a code expert explaining specific code entities. The user wants to understand a particular class, function, or component.

User's Question: '{question}'

Focus your analysis on:
- Purpose and responsibilities of the specific entity
- Input/output parameters and return types
- Internal logic and behavior
- Usage patterns and integration points

Structure your response with:
1. **Entity Purpose:** What this entity does and why it exists
2. **Interface Definition:** Parameters, return types, and public methods
3. **Internal Logic:** How it works internally
4. **Usage Context:** How and where it's used in the system
5. **Dependencies:** What it depends on and what depends on it

Retrieved Files: {file_references}""",

            'rationale': f"""You are a software architect explaining design decisions. The user wants to understand WHY certain choices were made in this codebase.

User's Question: '{question}'

Focus your analysis on:
- Design decisions and the reasoning behind them
- Trade-offs and alternatives considered
- Benefits and limitations of chosen approaches
- Context that influenced the decisions

Structure your response with:
1. **Decision Context:** What decision was made and the situation
2. **Reasoning:** Why this approach was chosen
3. **Trade-offs:** Benefits and drawbacks of the decision
4. **Alternatives:** Other options that could have been chosen
5. **Impact:** How this decision affects the overall system

Retrieved Files: {file_references}""",

            'data_flow': f"""You are a systems analyst explaining data flow and processing. The user wants to understand how data moves through this system.

User's Question: '{question}'

Focus your analysis on:
- Data flow paths and transformations
- Processing stages and data handling
- Input/output data structures
- Data persistence and state management

Structure your response with:
1. **Data Flow Overview:** High-level data movement through the system
2. **Data Sources:** Where data originates and how it enters
3. **Processing Stages:** How data is transformed and processed
4. **Data Storage:** How and where data is persisted
5. **Data Outputs:** Final data products and how they're used

Retrieved Files: {file_references}""",

            'general': f"""You are a senior software architect. The user has asked the following question: '{question}'

Based on the retrieved source files, provide a comprehensive analysis that directly answers the user's question.

Structure your response with:
1. **Direct Answer:** Address the user's question directly
2. **Core Components:** Key components relevant to the question
3. **Technical Details:** Important implementation details
4. **Context:** How this fits into the broader system
5. **Additional Insights:** Other relevant information

Retrieved Files: {file_references}"""
        }
        
        return prompts.get(question_type, prompts['general'])

    def get_architectural_analysis(self, question, documents):
        """Get architectural analysis based on retrieved documents with dynamic prompting."""
        file_references = " ".join([
            f"[{os.path.basename(doc.metadata['source'])}](<file-{os.path.basename(doc.metadata['source'])}>)"
            for doc in documents
        ])
        
        # Classify question and get appropriate prompt
        question_type = self._classify_question_type(question)
        prompt = self._get_prompt_for_question_type(question, question_type, file_references)

        payload = {
            "message_payload": {
                "messages": [{"content": prompt, "role": "USER"}],
                "chat_session_id": str(uuid.uuid4())
            }
        }
        
        files_data = {doc.metadata['source']: doc.page_content for doc in documents}
        
        return self._call_wca_api(payload, files_data)

    def summarize_file(self, file_path, file_content):
        """Summarize a single file's purpose and key components."""
        prompt = f"Provide a concise, one-paragraph summary of the following file's purpose and key components: [{os.path.basename(file_path)}](<file-{os.path.basename(file_path)}>)"
        
        payload = {
            "message_payload": {
                "messages": [{"content": prompt, "role": "USER"}],
                "chat_session_id": str(uuid.uuid4())
            }
        }
        
        files_data = {file_path: file_content}
        return self._call_wca_api(payload, files_data)

    def generate_hypothetical_questions(self, file_path, file_content):
        """Generate hypothetical questions that a file could help answer."""
        prompt = f"Based on the content of the file [{os.path.basename(file_path)}](<file-{os.path.basename(file_path)}>), generate a list of 3-5 high-level questions that this file could help answer. Return only the questions, each on a new line."
        
        payload = {
            "message_payload": {
                "messages": [{"content": prompt, "role": "USER"}],
                "chat_session_id": str(uuid.uuid4())
            }
        }
        
        files_data = {file_path: file_content}
        return self._call_wca_api(payload, files_data)

    def synthesize_summaries(self, summaries):
        """Synthesize multiple file summaries into architectural analysis."""
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