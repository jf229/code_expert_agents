#!/usr/bin/env python3
"""
Domain-Specialized Agents

Implements domain specialists that wrap existing agents with
specialized configurations and prompting for specific domains.
"""

import sys
import os
from typing import Dict, Any, List
from abc import ABC, abstractmethod

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing agents
from agents.top_k_retrieval import TopKAgent
from agents.iterate_and_synthesize import IterateAndSynthesizeAgent
from agents.graph_based_retrieval import GraphBasedAgent
from agents.multi_representation import MultiRepresentationAgent
from shared import UnifiedResponseGenerator


class DomainSpecialist(ABC):
    """Base class for domain-specialized agents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_agent = None
        self.domain_name = ""
        self.specialization_focus = ""
    
    @abstractmethod
    def analyze_question(self, question: str) -> str:
        """Analyze question with domain-specific expertise."""
        pass
    
    @abstractmethod
    def get_specialized_prompt(self, question: str) -> str:
        """Get domain-specific prompt enhancements."""
        pass
    
    def run_specialized_analysis(self, question: str) -> str:
        """Run analysis with domain specialization."""
        specialized_question = self.get_specialized_prompt(question)
        
        print(f"\n=== {self.domain_name} Specialist Analysis ===")
        print(f"Specialization: {self.specialization_focus}")
        print(f"Base Agent: {self.base_agent.__class__.__name__}")
        print("=" * 50)
        
        return self.base_agent.run(specialized_question)


class BackendSpecialist(DomainSpecialist):
    """Specialist for backend systems, APIs, and business logic."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_name = "Backend"
        self.specialization_focus = "APIs, business logic, services, data access"
        # Use graph-based agent for precise entity relationships
        self.base_agent = GraphBasedAgent()
    
    def analyze_question(self, question: str) -> str:
        """Analyze backend-focused questions."""
        backend_context = self._identify_backend_components(question)
        return self.run_specialized_analysis(question)
    
    def get_specialized_prompt(self, question: str) -> str:
        """Get backend-specific prompt enhancements."""
        backend_prompt = f"""
{question}

[BACKEND SPECIALIST CONTEXT]
Focus your analysis on backend systems and server-side architecture:

🔧 **Core Areas to Analyze:**
- API endpoints and service interfaces
- Business logic and domain models
- Data access layers and repositories
- Service orchestration and workflows
- Authentication and authorization systems
- Middleware and request processing
- Database interactions and ORM usage

🎯 **Backend-Specific Insights:**
- How does this component fit into the service architecture?
- What are the data flow patterns and business rules?
- How are external dependencies and integrations handled?
- What are the security and validation mechanisms?
- How does this relate to other backend services?

📋 **Technical Focus:**
- Controller and service layer interactions
- Database schema and query patterns
- API design and REST/GraphQL patterns
- Error handling and logging strategies
- Performance and scalability considerations
"""
        return backend_prompt
    
    def _identify_backend_components(self, question: str) -> Dict[str, Any]:
        """Identify backend-specific components mentioned in the question."""
        backend_keywords = {
            'api_related': ['api', 'endpoint', 'rest', 'graphql', 'service'],
            'business_logic': ['business', 'logic', 'rule', 'validation', 'workflow'],
            'data_access': ['repository', 'dao', 'orm', 'query', 'database'],
            'security': ['auth', 'authorization', 'authentication', 'security', 'permission'],
            'integration': ['integration', 'external', 'third-party', 'webhook']
        }
        
        found_components = {}
        question_lower = question.lower()
        
        for category, keywords in backend_keywords.items():
            found_components[category] = [kw for kw in keywords if kw in question_lower]
        
        return found_components


class FrontendSpecialist(DomainSpecialist):
    """Specialist for frontend components, UI, and user interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_name = "Frontend"
        self.specialization_focus = "UI components, user interactions, client-side logic"
        # Use top-k agent for quick component-focused retrieval
        self.base_agent = TopKAgent()
    
    def analyze_question(self, question: str) -> str:
        """Analyze frontend-focused questions."""
        frontend_context = self._identify_frontend_components(question)
        return self.run_specialized_analysis(question)
    
    def get_specialized_prompt(self, question: str) -> str:
        """Get frontend-specific prompt enhancements."""
        frontend_prompt = f"""
{question}

[FRONTEND SPECIALIST CONTEXT]
Focus your analysis on frontend systems and client-side architecture:

🎨 **Core Areas to Analyze:**
- UI components and component hierarchy
- User interaction patterns and event handling
- State management and data flow
- Routing and navigation systems
- Form handling and validation
- Styling and responsive design
- Client-side data fetching and caching

🎯 **Frontend-Specific Insights:**
- How does this component fit into the UI component tree?
- What are the user interaction flows and state changes?
- How is data passed between components (props/context)?
- What are the styling and design system patterns?
- How does this integrate with backend services?

📋 **Technical Focus:**
- Component lifecycle and rendering patterns
- Event handling and user input processing
- State management (Redux, Context, hooks)
- Performance optimization (lazy loading, memoization)
- Accessibility and user experience considerations
- Browser compatibility and responsive design
"""
        return frontend_prompt
    
    def _identify_frontend_components(self, question: str) -> Dict[str, Any]:
        """Identify frontend-specific components mentioned in the question."""
        frontend_keywords = {
            'components': ['component', 'widget', 'element', 'ui', 'interface'],
            'interactions': ['click', 'event', 'handler', 'input', 'form', 'button'],
            'state': ['state', 'redux', 'context', 'hook', 'store'],
            'styling': ['css', 'style', 'theme', 'design', 'responsive'],
            'frameworks': ['react', 'vue', 'angular', 'svelte', 'jsx', 'tsx']
        }
        
        found_components = {}
        question_lower = question.lower()
        
        for category, keywords in frontend_keywords.items():
            found_components[category] = [kw for kw in keywords if kw in question_lower]
        
        return found_components


class DataSpecialist(DomainSpecialist):
    """Specialist for data models, databases, and data flow."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_name = "Data"
        self.specialization_focus = "Data models, database design, data flow"
        # Use graph-based agent for relationship analysis
        self.base_agent = GraphBasedAgent()
    
    def analyze_question(self, question: str) -> str:
        """Analyze data-focused questions."""
        data_context = self._identify_data_components(question)
        return self.run_specialized_analysis(question)
    
    def get_specialized_prompt(self, question: str) -> str:
        """Get data-specific prompt enhancements."""
        data_prompt = f"""
{question}

[DATA SPECIALIST CONTEXT]
Focus your analysis on data architecture and information flow:

💾 **Core Areas to Analyze:**
- Data models and entity relationships
- Database schema and table design
- Data access patterns and queries
- Data validation and constraints
- Data flow between system components
- Caching strategies and performance
- Data migration and versioning

🎯 **Data-Specific Insights:**
- How is data structured and what are the relationships?
- What are the CRUD operations and query patterns?
- How does data flow through the application layers?
- What are the data consistency and integrity mechanisms?
- How is data cached and optimized for performance?

📋 **Technical Focus:**
- Entity-relationship modeling
- Database indexing and optimization
- Data transformation and serialization
- Backup and recovery strategies
- Data privacy and security measures
- Scalability and sharding considerations
"""
        return data_prompt
    
    def _identify_data_components(self, question: str) -> Dict[str, Any]:
        """Identify data-specific components mentioned in the question."""
        data_keywords = {
            'models': ['model', 'entity', 'schema', 'table', 'collection'],
            'operations': ['query', 'crud', 'select', 'insert', 'update', 'delete'],
            'relationships': ['relationship', 'foreign key', 'join', 'association'],
            'storage': ['database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql'],
            'performance': ['index', 'cache', 'optimization', 'performance', 'scaling']
        }
        
        found_components = {}
        question_lower = question.lower()
        
        for category, keywords in data_keywords.items():
            found_components[category] = [kw for kw in keywords if kw in question_lower]
        
        return found_components


class InfrastructureSpecialist(DomainSpecialist):
    """Specialist for infrastructure, deployment, and system configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_name = "Infrastructure"
        self.specialization_focus = "Deployment, configuration, tooling, DevOps"
        # Use iterate & synthesize for comprehensive system overview
        self.base_agent = IterateAndSynthesizeAgent()
    
    def analyze_question(self, question: str) -> str:
        """Analyze infrastructure-focused questions."""
        infra_context = self._identify_infrastructure_components(question)
        return self.run_specialized_analysis(question)
    
    def get_specialized_prompt(self, question: str) -> str:
        """Get infrastructure-specific prompt enhancements."""
        infra_prompt = f"""
{question}

[INFRASTRUCTURE SPECIALIST CONTEXT]
Focus your analysis on system infrastructure and operational concerns:

🏗️ **Core Areas to Analyze:**
- Deployment configurations and environments
- Build systems and CI/CD pipelines
- Container orchestration and Docker setup
- Environment configuration and secrets management
- Monitoring, logging, and observability
- Scaling and load balancing strategies
- Security and compliance configurations

🎯 **Infrastructure-Specific Insights:**
- How is the system deployed and configured?
- What are the build and deployment processes?
- How are different environments (dev/staging/prod) managed?
- What monitoring and alerting mechanisms are in place?
- How does the system handle scaling and reliability?

📋 **Technical Focus:**
- Dockerfiles and container configurations
- Kubernetes manifests and orchestration
- CI/CD pipeline definitions and workflows
- Configuration management and environment variables
- Infrastructure as Code (Terraform, CloudFormation)
- Service mesh and networking configurations
"""
        return infra_prompt
    
    def _identify_infrastructure_components(self, question: str) -> Dict[str, Any]:
        """Identify infrastructure-specific components mentioned in the question."""
        infra_keywords = {
            'deployment': ['deploy', 'deployment', 'release', 'environment', 'staging', 'production'],
            'containers': ['docker', 'container', 'kubernetes', 'k8s', 'pod', 'service'],
            'cicd': ['ci/cd', 'pipeline', 'build', 'test', 'automation', 'jenkins'],
            'monitoring': ['monitoring', 'logging', 'metrics', 'alerts', 'observability'],
            'config': ['config', 'configuration', 'environment', 'secrets', 'variables']
        }
        
        found_components = {}
        question_lower = question.lower()
        
        for category, keywords in infra_keywords.items():
            found_components[category] = [kw for kw in keywords if kw in question_lower]
        
        return found_components


class ArchitectureSpecialist(DomainSpecialist):
    """Specialist for high-level architecture and system design."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_name = "Architecture"
        self.specialization_focus = "System design, patterns, cross-cutting concerns"
        # Use multi-representation for comprehensive architectural analysis
        self.base_agent = MultiRepresentationAgent()
    
    def analyze_question(self, question: str) -> str:
        """Analyze architecture-focused questions."""
        arch_context = self._identify_architectural_components(question)
        return self.run_specialized_analysis(question)
    
    def get_specialized_prompt(self, question: str) -> str:
        """Get architecture-specific prompt enhancements."""
        arch_prompt = f"""
{question}

[ARCHITECTURE SPECIALIST CONTEXT]
Focus your analysis on high-level system architecture and design:

🏛️ **Core Areas to Analyze:**
- Overall system architecture and design patterns
- Component boundaries and interfaces
- Cross-cutting concerns (security, logging, caching)
- Architectural trade-offs and decisions
- Scalability and performance patterns
- Integration patterns and communication protocols
- Error handling and resilience strategies

🎯 **Architecture-Specific Insights:**
- What are the major architectural components and their responsibilities?
- How do different parts of the system communicate and integrate?
- What design patterns and architectural styles are employed?
- What are the key architectural decisions and their rationale?
- How does the architecture support non-functional requirements?

📋 **Technical Focus:**
- Service-oriented architecture and microservices patterns
- API design and integration strategies
- Data architecture and flow patterns
- Security architecture and access control
- Performance and scalability considerations
- Maintenance and evolution strategies
"""
        return arch_prompt
    
    def _identify_architectural_components(self, question: str) -> Dict[str, Any]:
        """Identify architecture-specific components mentioned in the question."""
        arch_keywords = {
            'patterns': ['pattern', 'design', 'architecture', 'mvc', 'mvp', 'mvvm'],
            'structure': ['component', 'module', 'layer', 'boundary', 'interface'],
            'communication': ['api', 'protocol', 'message', 'event', 'queue'],
            'quality': ['performance', 'scalability', 'security', 'reliability', 'maintainability'],
            'style': ['microservices', 'monolith', 'soa', 'serverless', 'event-driven']
        }
        
        found_components = {}
        question_lower = question.lower()
        
        for category, keywords in arch_keywords.items():
            found_components[category] = [kw for kw in keywords if kw in question_lower]
        
        return found_components


class SpecialistFactory:
    """Factory for creating domain specialists."""
    
    @staticmethod
    def create_specialist(domain: str, config: Dict[str, Any]) -> DomainSpecialist:
        """Create a specialist for the specified domain."""
        specialists = {
            'backend': BackendSpecialist,
            'frontend': FrontendSpecialist,
            'data': DataSpecialist,
            'infrastructure': InfrastructureSpecialist,
            'architecture': ArchitectureSpecialist
        }
        
        specialist_class = specialists.get(domain.lower())
        if not specialist_class:
            raise ValueError(f"Unknown domain: {domain}")
        
        return specialist_class(config)
    
    @staticmethod
    def get_available_domains() -> List[str]:
        """Get list of available domain specializations."""
        return ['backend', 'frontend', 'data', 'infrastructure', 'architecture']


def main():
    """Test domain specialists directly."""
    import argparse
    from shared_services import load_config, setup_and_pull_models
    
    parser = argparse.ArgumentParser(description="Domain-Specialized Code Analysis")
    parser.add_argument("domain", choices=SpecialistFactory.get_available_domains(), 
                       help="Domain specialist to use")
    parser.add_argument("question", type=str, help="Question to analyze")
    parser.add_argument("--repo", help="Repository path to analyze")
    args = parser.parse_args()
    
    # Set repository path if provided
    if args.repo:
        import os
        os.environ["REPO_PATH"] = args.repo
    
    # Load configuration and setup models
    config = load_config()
    setup_and_pull_models(config)
    
    # Create and run specialist
    specialist = SpecialistFactory.create_specialist(args.domain, config)
    result = specialist.analyze_question(args.question)
    
    print(f"\nDomain specialist analysis completed: {result}")


if __name__ == "__main__":
    main()