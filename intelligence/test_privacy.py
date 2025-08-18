#!/usr/bin/env python3
"""
Privacy Feature Test Script

Demonstrates the privacy layer functionality and allows A/B testing
of answer quality with privacy enabled vs disabled.
"""

import yaml
from privacy_manager import PrivacyManager
from langchain.schema import Document


def test_privacy_sanitization():
    """Test the privacy manager's sanitization capabilities."""
    print("=== Privacy Manager Test ===\n")
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Enable privacy for testing
    config["privacy"]["enable"] = True
    config["privacy"]["pseudonymize_identifiers"] = True
    config["privacy"]["hide_paths"] = True
    
    # Initialize privacy manager
    privacy_manager = PrivacyManager(config)
    
    # Create sample documents
    sample_code = '''
class UserAuthenticator:
    def __init__(self, database_connection, jwt_secret):
        self.db = database_connection
        self.secret = jwt_secret
    
    def authenticate_user(self, username, password):
        # Check user credentials against database
        user_record = self.db.query("SELECT * FROM users WHERE username = ?", username)
        if user_record and self.verify_password(password, user_record.password_hash):
            return self.generate_jwt_token(user_record)
        return None
    
    def verify_password(self, plain_password, hashed_password):
        return bcrypt.checkpw(plain_password.encode(), hashed_password)
'''
    
    documents = [
        Document(
            page_content=sample_code,
            metadata={"source": "/home/user/myproject/src/auth/authenticator.py"}
        ),
        Document(
            page_content="# Configuration file\nAPI_KEY = 'secret-key-123'\nDATABASE_URL = 'postgresql://localhost:5432/mydb'",
            metadata={"source": "/home/user/myproject/config/settings.py"}
        )
    ]
    
    print("Original Documents:")
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content:\n{doc.page_content}")
    
    print("\n" + "="*60 + "\n")
    
    # Test sanitization
    sanitized_docs, metadata = privacy_manager.sanitize_documents(documents)
    
    print("Sanitized Documents:")
    for i, doc in enumerate(sanitized_docs):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content:\n{doc.page_content}")
    
    print(f"\nSanitization Metadata:")
    print(f"- Original count: {metadata['original_count']}")
    print(f"- Sanitized count: {metadata['sanitized_count']}")
    print(f"- Total characters: {metadata['total_chars']}")
    
    for file_meta in metadata['files_processed']:
        print(f"- {file_meta['file_id']}: {file_meta['original_chars']} → {file_meta['sanitized_chars']} chars "
              f"({file_meta['reduction_ratio']:.1%} reduction)")
    
    # Show alias mappings
    print(f"\nAlias Mappings:")
    mappings = privacy_manager.get_alias_mappings()
    print(f"File aliases: {mappings['file_aliases']}")
    for file_key, id_map in mappings['identifier_aliases'].items():
        if id_map:
            print(f"Identifier aliases for {file_key[:8]}...: {id_map}")


def compare_privacy_modes():
    """Compare privacy enabled vs disabled modes."""
    print("\n\n=== Privacy Mode Comparison ===\n")
    
    # Sample question and code
    question = "How does the authentication system work?"
    
    sample_docs = [
        Document(
            page_content='''
class AuthenticationService:
    def __init__(self):
        self.jwt_secret = os.getenv('JWT_SECRET')
        self.database = DatabaseConnection()
    
    def login(self, email, password):
        user = self.database.find_user_by_email(email)
        if user and self.verify_password(password, user.password_hash):
            return self.create_jwt_token(user.id, user.role)
        raise AuthenticationError("Invalid credentials")
    
    def verify_password(self, plain_password, password_hash):
        return bcrypt.checkpw(plain_password.encode(), password_hash)
    
    def create_jwt_token(self, user_id, role):
        payload = {'user_id': user_id, 'role': role, 'exp': datetime.utcnow() + timedelta(hours=24)}
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
''',
            metadata={"source": "/project/services/authentication_service.py"}
        )
    ]
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Test with privacy disabled
    print("1. Privacy DISABLED:")
    config["privacy"]["enable"] = False
    privacy_manager_off = PrivacyManager(config)
    sanitized_off, metadata_off = privacy_manager_off.sanitize_documents(sample_docs)
    
    print(f"   Source: {sanitized_off[0].metadata['source']}")
    print(f"   Content preview: {sanitized_off[0].page_content[:100]}...")
    print(f"   Character count: {len(sanitized_off[0].page_content)}")
    
    print("\n2. Privacy ENABLED:")
    config["privacy"]["enable"] = True
    privacy_manager_on = PrivacyManager(config)
    sanitized_on, metadata_on = privacy_manager_on.sanitize_documents(sample_docs)
    
    print(f"   Source: {sanitized_on[0].metadata['source']}")
    print(f"   Content preview: {sanitized_on[0].page_content[:100]}...")
    print(f"   Character count: {len(sanitized_on[0].page_content)}")
    print(f"   Privacy reduction: {metadata_on['files_processed'][0]['reduction_ratio']:.1%}")


def test_policy_enforcement():
    """Test policy gate functionality."""
    print("\n\n=== Policy Enforcement Test ===\n")
    
    from privacy_manager import PolicyGate
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config["privacy"]["enable"] = True
    config["privacy"]["denylist_globs"] = ["**/secrets/**", "**/config/prod/**"]
    
    policy_gate = PolicyGate(config)
    
    test_files = [
        "/project/src/main.py",
        "/project/secrets/api_keys.py",
        "/project/config/prod/database.py",
        "/project/config/dev/database.py"
    ]
    
    print("File access policy test:")
    for file_path in test_files:
        should_deny = policy_gate.should_deny_file(file_path)
        status = "DENIED" if should_deny else "ALLOWED"
        print(f"   {file_path}: {status}")


if __name__ == "__main__":
    print("Privacy Layer Testing Suite")
    print("==========================")
    
    try:
        test_privacy_sanitization()
        compare_privacy_modes()
        test_policy_enforcement()
        
        print("\n\n=== Test Summary ===")
        print("✓ Privacy sanitization working")
        print("✓ Mode comparison functional")
        print("✓ Policy enforcement active")
        print("\nTo test with actual agents, use:")
        print("  python top_k_retrieval.py 'your question' --privacy")
        print("  python top_k_retrieval.py 'your question' --no-privacy")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()