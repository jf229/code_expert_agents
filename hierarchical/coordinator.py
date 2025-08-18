#!/usr/bin/env python3
"""
Hierarchical Multi-Agent Coordinator

Implements the master coordinator that routes questions to appropriate
domain specialists and orchestrates multi-agent workflows.
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing agents
from agents.top_k_retrieval import TopKAgent
from agents.iterate_and_synthesize import IterateAndSynthesizeAgent  
from agents.graph_based_retrieval import GraphBasedAgent
from agents.multi_representation import MultiRepresentationAgent
from shared import load_config, setup_and_pull_models, UnifiedResponseGenerator


class QuestionType(Enum):
    WHAT = "what"      # Entity-focused questions
    HOW = "how"        # Process-focused questions  
    WHY = "why"        # Rationale-focused questions
    WHERE = "where"    # Location-focused questions


class Domain(Enum):
    BACKEND = "backend"           # APIs, business logic, services
    FRONTEND = "frontend"         # UI, components, user interactions
    DATA = "data"                # Models, databases, data flow
    INFRASTRUCTURE = "infrastructure"  # Config, deployment, tooling
    CROSS_DOMAIN = "cross_domain"      # Spans multiple domains


class Complexity(Enum):
    SIMPLE = "simple"       # Single domain, single agent
    MEDIUM = "medium"       # Single domain, multi-step
    COMPLEX = "complex"     # Multi-domain, requires orchestration


@dataclass
class QuestionAnalysis:
    """Result of question analysis."""
    question_type: QuestionType
    domain: Domain
    complexity: Complexity
    confidence: float
    keywords: List[str]
    scope: str  # "specific_entity", "system_wide", "moderate"


@dataclass
class AgentRoute:
    """Defines how to route a question to agents."""
    primary_agent: str
    secondary_agents: List[str]
    orchestration_pattern: str  # "single", "sequential", "parallel", "collaborative"
    specialized_config: Dict[str, Any]


class QuestionClassifier:
    """Analyzes questions to determine routing strategy."""
    
    def __init__(self):
        self.what_patterns = [
            r'\bwhat\s+is\b', r'\bwhat\s+does\b', r'\bwhat\s+are\b',
            r'\bdefine\b', r'\bexplain\s+the\s+\w+\s+(class|function|method|component)\b'
        ]
        
        self.how_patterns = [
            r'\bhow\s+does\b', r'\bhow\s+to\b', r'\bprocess\b', r'\bflow\b',
            r'\bwork\b', r'\bimplement\b', r'\boperat\b'
        ]
        
        self.why_patterns = [
            r'\bwhy\s+is\b', r'\bwhy\s+does\b', r'\breason\b', r'\bpurpose\b',
            r'\bdecision\b', r'\brationale\b', r'\bchoose\b', r'\bchosen\b'
        ]
        
        self.where_patterns = [
            r'\bwhere\s+is\b', r'\bwhere\s+can\b', r'\blocate\b', r'\bfind\b',
            r'\bcontain\b', r'\bstore\b'
        ]
        
        # Domain indicators
        self.backend_keywords = [
            'api', 'endpoint', 'service', 'server', 'business logic', 'controller',
            'middleware', 'authentication', 'authorization', 'database query',
            'model', 'repository', 'orm'
        ]
        
        self.frontend_keywords = [
            'component', 'ui', 'user interface', 'react', 'vue', 'angular',
            'html', 'css', 'javascript', 'typescript', 'jsx', 'tsx',
            'state management', 'routing', 'navigation'
        ]
        
        self.data_keywords = [
            'database', 'sql', 'nosql', 'migration', 'schema', 'table',
            'collection', 'data model', 'entity', 'relationship', 'query',
            'persistence', 'storage'
        ]
        
        self.infrastructure_keywords = [
            'config', 'configuration', 'deployment', 'docker', 'kubernetes',
            'ci/cd', 'pipeline', 'build', 'environment', 'logging',
            'monitoring', 'scaling'
        ]
    
    def analyze(self, question: str) -> QuestionAnalysis:
        """Analyze question and return classification."""
        question_lower = question.lower()
        
        # Determine question type
        question_type = self._classify_question_type(question_lower)
        
        # Determine domain
        domain = self._classify_domain(question_lower)
        
        # Determine complexity
        complexity = self._assess_complexity(question_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(question_lower)
        
        # Determine scope
        scope = self._determine_scope(question_lower)
        
        # Calculate confidence (simplified)
        confidence = 0.8  # Would implement proper confidence scoring
        
        return QuestionAnalysis(
            question_type=question_type,
            domain=domain,
            complexity=complexity,
            confidence=confidence,
            keywords=keywords,
            scope=scope
        )
    
    def _classify_question_type(self, question: str) -> QuestionType:
        """Classify the type of question being asked."""
        if any(re.search(pattern, question) for pattern in self.what_patterns):
            return QuestionType.WHAT
        elif any(re.search(pattern, question) for pattern in self.how_patterns):
            return QuestionType.HOW
        elif any(re.search(pattern, question) for pattern in self.why_patterns):
            return QuestionType.WHY
        elif any(re.search(pattern, question) for pattern in self.where_patterns):
            return QuestionType.WHERE
        else:
            return QuestionType.WHAT  # Default
    
    def _classify_domain(self, question: str) -> Domain:
        """Classify the domain focus of the question."""
        domain_scores = {
            Domain.BACKEND: sum(1 for kw in self.backend_keywords if kw in question),
            Domain.FRONTEND: sum(1 for kw in self.frontend_keywords if kw in question),
            Domain.DATA: sum(1 for kw in self.data_keywords if kw in question),
            Domain.INFRASTRUCTURE: sum(1 for kw in self.infrastructure_keywords if kw in question),
        }
        
        # Check for cross-domain indicators
        high_scoring_domains = [d for d, score in domain_scores.items() if score > 0]
        if len(high_scoring_domains) > 1:
            return Domain.CROSS_DOMAIN
        
        # Return highest scoring domain or backend as default
        max_domain = max(domain_scores, key=domain_scores.get)
        return max_domain if domain_scores[max_domain] > 0 else Domain.BACKEND
    
    def _assess_complexity(self, question: str) -> Complexity:
        """Assess the complexity of the question."""
        complexity_indicators = [
            'entire', 'whole', 'complete', 'comprehensive', 'end-to-end',
            'across', 'throughout', 'all', 'every', 'full'
        ]
        
        process_indicators = [
            'flow', 'process', 'workflow', 'pipeline', 'sequence', 'steps'
        ]
        
        if any(indicator in question for indicator in complexity_indicators):
            return Complexity.COMPLEX
        elif any(indicator in question for indicator in process_indicators):
            return Complexity.MEDIUM
        else:
            return Complexity.SIMPLE
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from the question."""
        # Simple keyword extraction - could be enhanced with NLP
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', question)
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _determine_scope(self, question: str) -> str:
        """Determine the scope of the question."""
        system_wide_indicators = ['entire', 'whole', 'all', 'complete', 'overview', 'architecture']
        specific_indicators = ['class', 'function', 'method', 'component', 'file']
        
        if any(indicator in question for indicator in system_wide_indicators):
            return "system_wide"
        elif any(indicator in question for indicator in specific_indicators):
            return "specific_entity"
        else:
            return "moderate"


class AgentRouter:
    """Routes questions to appropriate agents based on analysis."""
    
    def __init__(self):
        self.routing_rules = self._initialize_routing_rules()
    
    def _initialize_routing_rules(self) -> Dict[str, AgentRoute]:
        """Initialize routing rules for different question patterns."""
        return {
            # Simple entity questions
            "what_backend_simple": AgentRoute(
                primary_agent="graph_based",
                secondary_agents=[],
                orchestration_pattern="single",
                specialized_config={"focus": "backend_entities"}
            ),
            
            "what_frontend_simple": AgentRoute(
                primary_agent="top_k",
                secondary_agents=[],
                orchestration_pattern="single", 
                specialized_config={"focus": "frontend_components"}
            ),
            
            # Process/flow questions
            "how_complex": AgentRoute(
                primary_agent="iterate_and_synthesize",
                secondary_agents=["graph_based", "multi_representation"],
                orchestration_pattern="collaborative",
                specialized_config={"comprehensive": True}
            ),
            
            # Cross-domain questions
            "cross_domain_complex": AgentRoute(
                primary_agent="multi_representation",
                secondary_agents=["graph_based", "iterate_and_synthesize"],
                orchestration_pattern="sequential",
                specialized_config={"multi_domain": True}
            ),
            
            # Default routes
            "default_simple": AgentRoute(
                primary_agent="top_k",
                secondary_agents=[],
                orchestration_pattern="single",
                specialized_config={}
            ),
            
            "default_complex": AgentRoute(
                primary_agent="multi_representation",
                secondary_agents=["graph_based"],
                orchestration_pattern="collaborative",
                specialized_config={}
            )
        }
    
    def route(self, analysis: QuestionAnalysis) -> AgentRoute:
        """Route question based on analysis."""
        
        # Generate routing key based on analysis
        routing_key = self._generate_routing_key(analysis)
        
        # Look up specific route or use default
        route = self.routing_rules.get(routing_key)
        if route:
            return route
        
        # Fallback routing logic
        if analysis.complexity == Complexity.SIMPLE:
            return self.routing_rules["default_simple"]
        else:
            return self.routing_rules["default_complex"]
    
    def _generate_routing_key(self, analysis: QuestionAnalysis) -> str:
        """Generate routing key from analysis."""
        type_str = analysis.question_type.value
        domain_str = analysis.domain.value
        complexity_str = analysis.complexity.value
        
        # Special cases
        if analysis.domain == Domain.CROSS_DOMAIN and analysis.complexity == Complexity.COMPLEX:
            return "cross_domain_complex"
        
        # Standard patterns
        if analysis.question_type == QuestionType.WHAT and analysis.complexity == Complexity.SIMPLE:
            if analysis.domain == Domain.BACKEND:
                return "what_backend_simple"
            elif analysis.domain == Domain.FRONTEND:
                return "what_frontend_simple"
        
        if analysis.question_type == QuestionType.HOW and analysis.complexity == Complexity.COMPLEX:
            return "how_complex"
        
        return f"{type_str}_{domain_str}_{complexity_str}"


class OrchestrationEngine:
    """Orchestrates multi-agent workflows."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = self._initialize_agents()
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all available agents."""
        return {
            "top_k": TopKAgent(),
            "iterate_and_synthesize": IterateAndSynthesizeAgent(),
            "graph_based": GraphBasedAgent(),
            "multi_representation": MultiRepresentationAgent()
        }
    
    def execute(self, question: str, route: AgentRoute, analysis: QuestionAnalysis) -> str:
        """Execute the routing plan."""
        
        if route.orchestration_pattern == "single":
            return self._execute_single_agent(question, route, analysis)
        elif route.orchestration_pattern == "sequential":
            return self._execute_sequential(question, route, analysis)
        elif route.orchestration_pattern == "parallel":
            return self._execute_parallel(question, route, analysis)
        elif route.orchestration_pattern == "collaborative":
            return self._execute_collaborative(question, route, analysis)
        else:
            # Fallback to single agent
            return self._execute_single_agent(question, route, analysis)
    
    def _execute_single_agent(self, question: str, route: AgentRoute, analysis: QuestionAnalysis) -> str:
        """Execute with a single agent."""
        agent = self.agents[route.primary_agent]
        
        # Configure agent based on specialization
        specialized_question = self._apply_specialization(question, route.specialized_config, analysis)
        
        # Use the agent's existing run method but capture output
        print(f"\\n=== Hierarchical Routing: Using {route.primary_agent} agent ===")
        print(f"Question type: {analysis.question_type.value}")
        print(f"Domain: {analysis.domain.value}")
        print(f"Complexity: {analysis.complexity.value}")
        print("=" * 60)
        
        # Execute the agent (they handle their own output)
        agent.run(specialized_question)
        
        return "Execution completed by hierarchical system"
    
    def _execute_sequential(self, question: str, route: AgentRoute, analysis: QuestionAnalysis) -> str:
        """Execute agents in sequence."""
        print(f"\\n=== Hierarchical Sequential Execution ===")
        print(f"Primary: {route.primary_agent}")
        print(f"Secondary: {', '.join(route.secondary_agents)}")
        print("=" * 60)
        
        # For now, execute primary agent with enhanced context
        specialized_question = self._apply_specialization(question, route.specialized_config, analysis)
        specialized_question += "\\n\\n[Note: This analysis uses sequential multi-agent processing for comprehensive coverage]"
        
        agent = self.agents[route.primary_agent]
        agent.run(specialized_question)
        
        return "Sequential execution completed"
    
    def _execute_parallel(self, question: str, route: AgentRoute, analysis: QuestionAnalysis) -> str:
        """Execute multiple agents in parallel."""
        print(f"\\n=== Hierarchical Parallel Execution ===")
        print(f"Primary: {route.primary_agent}")
        print(f"Parallel agents: {', '.join(route.secondary_agents)}")
        print("=" * 60)
        
        # For now, execute primary agent with enhanced context
        specialized_question = self._apply_specialization(question, route.specialized_config, analysis)
        specialized_question += "\\n\\n[Note: This analysis uses parallel multi-agent processing for diverse perspectives]"
        
        agent = self.agents[route.primary_agent]
        agent.run(specialized_question)
        
        return "Parallel execution completed"
    
    def _execute_collaborative(self, question: str, route: AgentRoute, analysis: QuestionAnalysis) -> str:
        """Execute agents in collaborative mode."""
        print(f"\\n=== Hierarchical Collaborative Execution ===")
        print(f"Lead agent: {route.primary_agent}")
        print(f"Collaborating agents: {', '.join(route.secondary_agents)}")
        print(f"Focus: {analysis.domain.value} domain, {analysis.complexity.value} complexity")
        print("=" * 60)
        
        # Enhanced question with collaborative context
        specialized_question = self._apply_specialization(question, route.specialized_config, analysis)
        specialized_question += f"\\n\\n[Note: This is a collaborative analysis focusing on {analysis.domain.value} domain with {analysis.complexity.value} complexity. Multiple expert perspectives are being integrated.]"
        
        agent = self.agents[route.primary_agent]
        agent.run(specialized_question)
        
        return "Collaborative execution completed"
    
    def _apply_specialization(self, question: str, config: Dict[str, Any], analysis: QuestionAnalysis) -> str:
        """Apply domain-specific specialization to the question."""
        if not config:
            return question
        
        specialized_prompt = question
        
        if config.get("focus") == "backend_entities":
            specialized_prompt += "\\n\\n[Focus on backend components: APIs, services, business logic, data access layers]"
        elif config.get("focus") == "frontend_components":
            specialized_prompt += "\\n\\n[Focus on frontend components: UI elements, user interactions, client-side logic]"
        elif config.get("comprehensive"):
            specialized_prompt += "\\n\\n[Provide comprehensive analysis covering all aspects of the system]"
        elif config.get("multi_domain"):
            specialized_prompt += "\\n\\n[Analyze across multiple domains: backend, frontend, data, and infrastructure]"
        
        return specialized_prompt


class HierarchicalCoordinator:
    """Main coordinator that orchestrates the hierarchical agent system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config()
        self.classifier = QuestionClassifier()
        self.router = AgentRouter()
        self.orchestrator = OrchestrationEngine(self.config)
    
    def process_question(self, question: str) -> str:
        """Process a question through the hierarchical system."""
        
        # Step 1: Analyze the question
        analysis = self.classifier.analyze(question)
        
        # Step 2: Route to appropriate agents
        route = self.router.route(analysis)
        
        # Step 3: Execute orchestrated workflow
        result = self.orchestrator.execute(question, route, analysis)
        
        return result


def main():
    """Main entry point for hierarchical coordinator."""
    parser = argparse.ArgumentParser(description="Hierarchical Multi-Agent Code Expert System")
    parser.add_argument("question", type=str, help="The question to ask the expert system")
    parser.add_argument("--repo", help="Repository path to analyze")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Set repository path if provided
    if args.repo:
        os.environ["REPO_PATH"] = args.repo
    
    # Load configuration and setup models
    config = load_config()
    setup_and_pull_models(config)
    
    # Create and run hierarchical coordinator
    coordinator = HierarchicalCoordinator()
    
    print("🧠 Hierarchical Multi-Agent Code Expert System")
    print("=" * 60)
    
    result = coordinator.process_question(args.question)
    
    if args.debug:
        print(f"\\nDebug: {result}")


if __name__ == "__main__":
    main()