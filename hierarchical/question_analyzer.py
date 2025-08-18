#!/usr/bin/env python3
"""
Enhanced Question Analyzer

Implements advanced question analysis that leverages existing storage
systems (vector stores, graphs, multi-representations) to optimize
retrieval strategies and agent selection.
"""

import re
import os
import pickle
import sys
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_config


class StorageType(Enum):
    """Types of existing storage systems."""
    VECTOR_STORE = "vector_store"
    CODE_GRAPH = "code_graph"
    MULTI_REPRESENTATIONS = "multi_representations"
    RAW_DOCUMENTS = "raw_documents"


@dataclass
class StorageAvailability:
    """Tracks what storage systems are available."""
    vector_store: bool = False
    code_graph: bool = False
    multi_representations: bool = False
    raw_documents: bool = False


@dataclass
class QuestionComplexity:
    """Detailed complexity analysis of a question."""
    scope_complexity: float  # 0-1, system-wide vs specific
    technical_complexity: float  # 0-1, simple vs advanced concepts
    relationship_complexity: float  # 0-1, single entity vs multi-entity
    process_complexity: float  # 0-1, static info vs dynamic processes
    overall_complexity: float  # 0-1, overall complexity score


@dataclass
class OptimalStrategy:
    """Optimal strategy recommendation based on analysis."""
    primary_storage: StorageType
    secondary_storage: Optional[StorageType]
    agent_recommendation: str
    confidence: float
    reasoning: str


class EnhancedQuestionAnalyzer:
    """Advanced question analyzer that considers available storage systems."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config()
        self.storage_config = self.config.get("storage", {})
        self.storage_availability = self._check_storage_availability()
        
        # Pattern definitions for question analysis
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for question analysis."""
        
        # Scope indicators
        self.system_wide_patterns = [
            r'\b(entire|whole|all|complete|full|overall|comprehensive)\b',
            r'\b(architecture|system|application|project)\b',
            r'\b(end-to-end|across|throughout)\b'
        ]
        
        self.specific_entity_patterns = [
            r'\b(class|function|method|component|module|file)\s+\w+\b',
            r'\bwhat\s+is\s+\w+\b',
            r'\b(specific|particular|individual)\b'
        ]
        
        # Technical complexity indicators
        self.high_technical_patterns = [
            r'\b(algorithm|performance|optimization|scaling)\b',
            r'\b(security|authentication|authorization)\b',
            r'\b(concurrency|threading|async|parallel)\b',
            r'\b(design\s+pattern|architectural\s+pattern)\b'
        ]
        
        # Relationship complexity indicators
        self.multi_entity_patterns = [
            r'\b(interact|relationship|depend|connect|integrate)\b',
            r'\b(flow|process|workflow|pipeline)\b',
            r'\b(between|among|across)\b.*\b(component|service|module)\b'
        ]
        
        # Process complexity indicators
        self.process_patterns = [
            r'\bhow\s+(does|do|is|are)\b',
            r'\b(process|workflow|flow|sequence|steps)\b',
            r'\b(implement|execute|perform|handle)\b'
        ]
    
    def _check_storage_availability(self) -> StorageAvailability:
        """Check which storage systems are available."""
        availability = StorageAvailability()
        
        # Check vector store
        vector_store_path = self.storage_config.get("vector_store", "vector_store")
        availability.vector_store = os.path.exists(vector_store_path)
        
        # Check code graph
        graph_path = self.storage_config.get("code_graph", "code_graph.gpickle")
        availability.code_graph = os.path.exists(graph_path)
        
        # Check multi-representations
        multi_rep_path = self.storage_config.get("multi_representations", "multi_representations.pkl")
        availability.multi_representations = os.path.exists(multi_rep_path)
        
        # Check raw documents
        raw_docs_path = self.storage_config.get("raw_docs", "raw_documents.pkl")
        availability.raw_documents = os.path.exists(raw_docs_path)
        
        return availability
    
    def analyze_question_complexity(self, question: str) -> QuestionComplexity:
        """Analyze the complexity dimensions of a question."""
        question_lower = question.lower()
        
        # Scope complexity
        scope_complexity = self._calculate_scope_complexity(question_lower)
        
        # Technical complexity
        technical_complexity = self._calculate_technical_complexity(question_lower)
        
        # Relationship complexity
        relationship_complexity = self._calculate_relationship_complexity(question_lower)
        
        # Process complexity
        process_complexity = self._calculate_process_complexity(question_lower)
        
        # Overall complexity (weighted average)
        overall_complexity = (
            scope_complexity * 0.3 +
            technical_complexity * 0.2 +
            relationship_complexity * 0.3 +
            process_complexity * 0.2
        )
        
        return QuestionComplexity(
            scope_complexity=scope_complexity,
            technical_complexity=technical_complexity,
            relationship_complexity=relationship_complexity,
            process_complexity=process_complexity,
            overall_complexity=overall_complexity
        )
    
    def _calculate_scope_complexity(self, question: str) -> float:
        """Calculate scope complexity (0=specific, 1=system-wide)."""
        system_wide_score = sum(1 for pattern in self.system_wide_patterns 
                               if re.search(pattern, question))
        specific_score = sum(1 for pattern in self.specific_entity_patterns 
                            if re.search(pattern, question))
        
        if system_wide_score > specific_score:
            return min(0.8, 0.3 + system_wide_score * 0.2)
        elif specific_score > 0:
            return max(0.2, 0.5 - specific_score * 0.1)
        else:
            return 0.5  # Medium complexity if unclear
    
    def _calculate_technical_complexity(self, question: str) -> float:
        """Calculate technical complexity (0=simple, 1=advanced)."""
        high_tech_score = sum(1 for pattern in self.high_technical_patterns 
                             if re.search(pattern, question))
        
        return min(1.0, 0.3 + high_tech_score * 0.2)
    
    def _calculate_relationship_complexity(self, question: str) -> float:
        """Calculate relationship complexity (0=single entity, 1=multi-entity)."""
        multi_entity_score = sum(1 for pattern in self.multi_entity_patterns 
                                if re.search(pattern, question))
        
        return min(1.0, 0.2 + multi_entity_score * 0.3)
    
    def _calculate_process_complexity(self, question: str) -> float:
        """Calculate process complexity (0=static, 1=dynamic process)."""
        process_score = sum(1 for pattern in self.process_patterns 
                           if re.search(pattern, question))
        
        return min(1.0, 0.1 + process_score * 0.3)
    
    def recommend_optimal_strategy(self, question: str) -> OptimalStrategy:
        """Recommend optimal strategy based on question and available storage."""
        complexity = self.analyze_question_complexity(question)
        
        # Strategy selection logic based on complexity and available storage
        if complexity.scope_complexity > 0.7:
            # System-wide questions
            return self._recommend_for_system_wide(complexity)
        elif complexity.relationship_complexity > 0.6:
            # Relationship-focused questions
            return self._recommend_for_relationships(complexity)
        elif complexity.process_complexity > 0.6:
            # Process-focused questions  
            return self._recommend_for_processes(complexity)
        else:
            # Specific entity questions
            return self._recommend_for_specific_entities(complexity)
    
    def _recommend_for_system_wide(self, complexity: QuestionComplexity) -> OptimalStrategy:
        """Recommend strategy for system-wide questions."""
        if self.storage_availability.raw_documents:
            return OptimalStrategy(
                primary_storage=StorageType.RAW_DOCUMENTS,
                secondary_storage=StorageType.MULTI_REPRESENTATIONS if self.storage_availability.multi_representations else None,
                agent_recommendation="iterate_and_synthesize",
                confidence=0.9,
                reasoning="System-wide question best handled by comprehensive document analysis"
            )
        elif self.storage_availability.multi_representations:
            return OptimalStrategy(
                primary_storage=StorageType.MULTI_REPRESENTATIONS,
                secondary_storage=None,
                agent_recommendation="multi_representation",
                confidence=0.8,
                reasoning="Multi-representations available for comprehensive analysis"
            )
        else:
            return self._fallback_strategy("No comprehensive storage available")
    
    def _recommend_for_relationships(self, complexity: QuestionComplexity) -> OptimalStrategy:
        """Recommend strategy for relationship-focused questions."""
        if self.storage_availability.code_graph:
            return OptimalStrategy(
                primary_storage=StorageType.CODE_GRAPH,
                secondary_storage=StorageType.VECTOR_STORE if self.storage_availability.vector_store else None,
                agent_recommendation="graph_based",
                confidence=0.95,
                reasoning="Graph storage optimal for relationship analysis"
            )
        elif self.storage_availability.multi_representations:
            return OptimalStrategy(
                primary_storage=StorageType.MULTI_REPRESENTATIONS,
                secondary_storage=None,
                agent_recommendation="multi_representation",
                confidence=0.7,
                reasoning="Multi-representations can handle relationships, but graph would be better"
            )
        else:
            return self._fallback_strategy("No relationship-optimized storage available")
    
    def _recommend_for_processes(self, complexity: QuestionComplexity) -> OptimalStrategy:
        """Recommend strategy for process-focused questions."""
        if complexity.scope_complexity > 0.5:
            # Complex processes spanning multiple components
            if self.storage_availability.raw_documents:
                return OptimalStrategy(
                    primary_storage=StorageType.RAW_DOCUMENTS,
                    secondary_storage=StorageType.CODE_GRAPH if self.storage_availability.code_graph else None,
                    agent_recommendation="iterate_and_synthesize",
                    confidence=0.85,
                    reasoning="Complex processes require comprehensive analysis"
                )
        
        # Simpler processes
        if self.storage_availability.vector_store:
            return OptimalStrategy(
                primary_storage=StorageType.VECTOR_STORE,
                secondary_storage=StorageType.CODE_GRAPH if self.storage_availability.code_graph else None,
                agent_recommendation="top_k",
                confidence=0.8,
                reasoning="Vector store good for focused process analysis"
            )
        else:
            return self._fallback_strategy("No optimized storage for process analysis")
    
    def _recommend_for_specific_entities(self, complexity: QuestionComplexity) -> OptimalStrategy:
        """Recommend strategy for specific entity questions."""
        if self.storage_availability.code_graph:
            return OptimalStrategy(
                primary_storage=StorageType.CODE_GRAPH,
                secondary_storage=None,
                agent_recommendation="graph_based",
                confidence=0.9,
                reasoning="Graph storage optimal for specific entity analysis"
            )
        elif self.storage_availability.vector_store:
            return OptimalStrategy(
                primary_storage=StorageType.VECTOR_STORE,
                secondary_storage=None,
                agent_recommendation="top_k",
                confidence=0.85,
                reasoning="Vector store good for specific entity retrieval"
            )
        else:
            return self._fallback_strategy("No entity-optimized storage available")
    
    def _fallback_strategy(self, reason: str) -> OptimalStrategy:
        """Provide fallback strategy when optimal storage isn't available."""
        # Find any available storage
        if self.storage_availability.vector_store:
            primary = StorageType.VECTOR_STORE
            agent = "top_k"
        elif self.storage_availability.raw_documents:
            primary = StorageType.RAW_DOCUMENTS
            agent = "iterate_and_synthesize"
        elif self.storage_availability.multi_representations:
            primary = StorageType.MULTI_REPRESENTATIONS
            agent = "multi_representation"
        elif self.storage_availability.code_graph:
            primary = StorageType.CODE_GRAPH
            agent = "graph_based"
        else:
            # No storage available - this shouldn't happen in normal usage
            primary = StorageType.RAW_DOCUMENTS
            agent = "top_k"
            reason += " - No storage systems found, will run data ingestion"
        
        return OptimalStrategy(
            primary_storage=primary,
            secondary_storage=None,
            agent_recommendation=agent,
            confidence=0.5,
            reasoning=f"Fallback strategy: {reason}"
        )
    
    def get_storage_optimization_suggestions(self) -> Dict[str, str]:
        """Get suggestions for optimizing storage based on availability."""
        suggestions = {}
        
        if not self.storage_availability.vector_store:
            suggestions["vector_store"] = "Run any agent to create vector store for fast entity retrieval"
        
        if not self.storage_availability.code_graph:
            suggestions["code_graph"] = "Run 'python graph_based_retrieval.py --build-graph' for relationship analysis"
        
        if not self.storage_availability.multi_representations:
            suggestions["multi_representations"] = "Run 'python multi_representation.py --build-representations' for adaptive strategies"
        
        if not self.storage_availability.raw_documents:
            suggestions["raw_documents"] = "Data ingestion will run automatically when needed"
        
        return suggestions
    
    def analyze_storage_for_question(self, question: str) -> Dict[str, Any]:
        """Comprehensive analysis of question with storage recommendations."""
        complexity = self.analyze_question_complexity(question)
        strategy = self.recommend_optimal_strategy(question)
        suggestions = self.get_storage_optimization_suggestions()
        
        return {
            "question": question,
            "complexity_analysis": {
                "scope": complexity.scope_complexity,
                "technical": complexity.technical_complexity,
                "relationships": complexity.relationship_complexity,
                "processes": complexity.process_complexity,
                "overall": complexity.overall_complexity
            },
            "storage_availability": {
                "vector_store": self.storage_availability.vector_store,
                "code_graph": self.storage_availability.code_graph,
                "multi_representations": self.storage_availability.multi_representations,
                "raw_documents": self.storage_availability.raw_documents
            },
            "optimal_strategy": {
                "primary_storage": strategy.primary_storage.value,
                "secondary_storage": strategy.secondary_storage.value if strategy.secondary_storage else None,
                "recommended_agent": strategy.agent_recommendation,
                "confidence": strategy.confidence,
                "reasoning": strategy.reasoning
            },
            "optimization_suggestions": suggestions
        }


def main():
    """Test the enhanced question analyzer."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Enhanced Question Analysis")
    parser.add_argument("question", type=str, help="Question to analyze")
    parser.add_argument("--repo", help="Repository path to analyze")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    args = parser.parse_args()
    
    # Set repository path if provided
    if args.repo:
        os.environ["REPO_PATH"] = args.repo
    
    # Create analyzer and analyze question
    analyzer = EnhancedQuestionAnalyzer()
    analysis = analyzer.analyze_storage_for_question(args.question)
    
    if args.detailed:
        print(json.dumps(analysis, indent=2))
    else:
        strategy = analysis["optimal_strategy"]
        print(f"Recommended Agent: {strategy['recommended_agent']}")
        print(f"Primary Storage: {strategy['primary_storage']}")
        print(f"Confidence: {strategy['confidence']:.2f}")
        print(f"Reasoning: {strategy['reasoning']}")
        
        if analysis["optimization_suggestions"]:
            print("\nOptimization Suggestions:")
            for storage, suggestion in analysis["optimization_suggestions"].items():
                print(f"  {storage}: {suggestion}")


if __name__ == "__main__":
    main()