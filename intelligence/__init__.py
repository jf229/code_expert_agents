"""Intelligence and privacy features for the RAG system."""

from .privacy_manager import PrivacyManager, PolicyGate

try:
    from .repository_intelligence import RepositoryIntelligence
except ImportError:
    RepositoryIntelligence = None

try:
    from .workspace_manager import WorkspaceManager
except ImportError:
    WorkspaceManager = None

__all__ = [
    'PrivacyManager',
    'PolicyGate', 
    'RepositoryIntelligence',
    'WorkspaceManager'
]