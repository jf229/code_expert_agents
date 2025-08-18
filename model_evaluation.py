#!/usr/bin/env python3
"""
Ollama Model Evaluation Suite

Comprehensive evaluation of different Ollama models for code analysis tasks.
Tests each available model across various question types and provides
detailed performance comparisons.

Usage:
    python model_evaluation.py                    # Test all available models
    python model_evaluation.py --model deepseek-r1:8b  # Test specific model
    python model_evaluation.py --quick            # Run quick test (fewer questions)
"""

import os
import sys
import argparse
import time
import json
import subprocess
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import yaml

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.top_k_retrieval import TopKAgent
from shared import load_config


@dataclass
class ModelResult:
    """Results for a single model evaluation."""
    model_name: str
    total_questions: int
    successful_responses: int
    total_time: float
    avg_response_time: float
    response_quality_scores: List[float]
    responses: Dict[str, Any]
    errors: List[str]


class ModelEvaluator:
    """Evaluates different Ollama models for code analysis performance."""
    
    def __init__(self, repo_path: str = "/home/clutchcoder/working/rh_web/", quick_test: bool = False):
        self.repo_path = repo_path
        self.quick_test = quick_test
        self.results = {}
        
        # Test questions designed to evaluate different aspects of code analysis
        self.test_questions = [
            # Authentication/Security Questions
            "How does authentication work in this application?",
            "What login mechanisms are implemented?",
            
            # API/Architecture Questions  
            "What API endpoints are available?",
            "How is the Flask application structured?",
            
            # Code Understanding Questions
            "What does the fetch_and_process_option_orders function do?",
            "How are static files handled?",
            
            # Data Processing Questions
            "How is option data processed and filtered?",
            "What data structures are used for orders?",
        ]
        
        if quick_test:
            self.test_questions = self.test_questions[:4]  # Use only first 4 questions
        
        # Quality evaluation criteria (keywords that indicate good responses)
        self.quality_indicators = {
            'technical_accuracy': ['function', 'method', 'class', 'variable', 'import', 'return'],
            'code_understanding': ['authentication', 'login', 'flask', 'api', 'endpoint', 'route'],
            'detailed_analysis': ['because', 'when', 'if', 'then', 'however', 'therefore'],
            'structure': ['first', 'second', 'then', 'finally', 'steps', 'process'],
            'security_awareness': ['security', 'credential', 'password', 'token', 'auth']
        }
        
        print(f"🧪 Model Evaluation Suite Initialized")
        print(f"📁 Repository: {self.repo_path}")
        print(f"❓ Questions: {len(self.test_questions)}")
        print(f"⚡ Quick mode: {'Yes' if quick_test else 'No'}")
        print("=" * 70)
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            result = subprocess.run(['ollama', 'ls'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    # Filter out embedding models and vision models for LLM testing
                    if 'embed' not in model_name.lower() and 'vision' not in model_name.lower():
                        models.append(model_name)
            
            return models
        except subprocess.CalledProcessError as e:
            print(f"❌ Error getting Ollama models: {e}")
            return []
    
    def evaluate_response_quality(self, response: str) -> float:
        """Evaluate response quality based on various criteria."""
        if not response or len(response) < 50:
            return 0.0
        
        response_lower = response.lower()
        total_score = 0.0
        max_score = 0.0
        
        for category, indicators in self.quality_indicators.items():
            category_score = 0.0
            for indicator in indicators:
                if indicator in response_lower:
                    category_score += 1.0
            
            # Normalize by number of indicators in category
            category_score = min(category_score / len(indicators), 1.0)
            total_score += category_score
            max_score += 1.0
        
        # Add bonus for length (comprehensive responses)
        length_bonus = min(len(response) / 1000, 0.5)  # Up to 0.5 bonus
        total_score += length_bonus
        max_score += 0.5
        
        # Add bonus for code snippets or technical details
        code_bonus = 0.0
        if '```' in response or 'def ' in response or 'class ' in response:
            code_bonus = 0.3
        total_score += code_bonus
        max_score += 0.3
        
        return min(total_score / max_score, 1.0)
    
    def test_model(self, model_name: str) -> ModelResult:
        """Test a specific model with all questions."""
        print(f"\n🤖 Testing Model: {model_name}")
        print("-" * 50)
        
        # Update config to use this model
        config = load_config()
        original_model = config["llm"]["models"]["ollama"]
        config["llm"]["models"]["ollama"] = model_name
        
        # Save temporary config
        with open("temp_config.yaml", "w") as f:
            yaml.dump(config, f)
        
        successful_responses = 0
        total_time = 0.0
        quality_scores = []
        responses = {}
        errors = []
        
        # Set environment variable for repo
        os.environ["REPO_PATH"] = self.repo_path
        
        try:
            for i, question in enumerate(self.test_questions, 1):
                print(f"  📋 Question {i}/{len(self.test_questions)}: {question[:50]}...")
                
                start_time = time.time()
                try:
                    # Create fresh agent for each test to avoid state issues
                    agent = TopKAgent()
                    agent.config["llm"]["models"]["ollama"] = model_name
                    
                    # Capture response
                    import io
                    from contextlib import redirect_stdout, redirect_stderr
                    
                    captured_output = io.StringIO()
                    captured_errors = io.StringIO()
                    
                    with redirect_stdout(captured_output), redirect_stderr(captured_errors):
                        agent.run(question)
                    
                    response_time = time.time() - start_time
                    output = captured_output.getvalue()
                    error_output = captured_errors.getvalue()
                    
                    # Extract actual response from output (after "Answer:" section)
                    actual_response = ""
                    if "Answer:" in output:
                        actual_response = output.split("Answer:")[1].split("Source Documents:")[0].strip()
                    
                    if actual_response and len(actual_response) > 50:
                        successful_responses += 1
                        quality_score = self.evaluate_response_quality(actual_response)
                        quality_scores.append(quality_score)
                        
                        responses[question] = {
                            "response": actual_response[:500] + "..." if len(actual_response) > 500 else actual_response,
                            "response_time": response_time,
                            "quality_score": quality_score,
                            "word_count": len(actual_response.split())
                        }
                        
                        print(f"    ✅ Success ({response_time:.1f}s, quality: {quality_score:.2f})")
                    else:
                        error_msg = f"No valid response generated"
                        if error_output:
                            error_msg += f": {error_output[:100]}"
                        errors.append(f"Q{i}: {error_msg}")
                        print(f"    ❌ Failed: {error_msg}")
                    
                    total_time += response_time
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    total_time += response_time
                    error_msg = str(e)[:100]
                    errors.append(f"Q{i}: {error_msg}")
                    print(f"    ❌ Error ({response_time:.1f}s): {error_msg}")
                
                # Clean up cache between questions to avoid interference
                self.cleanup_cache()
        
        finally:
            # Restore original config
            config["llm"]["models"]["ollama"] = original_model
            with open("temp_config.yaml", "w") as f:
                yaml.dump(config, f)
            
            # Clean up temp config
            if os.path.exists("temp_config.yaml"):
                os.remove("temp_config.yaml")
        
        avg_response_time = total_time / len(self.test_questions) if self.test_questions else 0
        
        return ModelResult(
            model_name=model_name,
            total_questions=len(self.test_questions),
            successful_responses=successful_responses,
            total_time=total_time,
            avg_response_time=avg_response_time,
            response_quality_scores=quality_scores,
            responses=responses,
            errors=errors
        )
    
    def cleanup_cache(self):
        """Clean up cache files and vector database between tests."""
        print("    🧹 Cleaning cache...")
        
        # Remove pickle cache files
        cache_files = [
            "raw_documents.pkl", 
            "docstore.pkl", 
            "code_graph.gpickle",
            "multi_representations.pkl",
            "repository_metadata.pkl"
        ]
        
        for file in cache_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"      ✓ Removed {file}")
                except Exception as e:
                    print(f"      ⚠️ Could not remove {file}: {e}")
        
        # Remove vector store directory completely
        if os.path.exists("vector_store"):
            import shutil
            try:
                shutil.rmtree("vector_store")
                print("      ✓ Removed vector_store directory")
            except Exception as e:
                print(f"      ⚠️ Could not remove vector_store: {e}")
        
        # Remove any ChromaDB lock files or temp files
        import glob
        chroma_files = glob.glob("*.sqlite*") + glob.glob("*.db-*") + glob.glob("chroma.log")
        for file in chroma_files:
            try:
                os.remove(file)
                print(f"      ✓ Removed ChromaDB file: {file}")
            except Exception as e:
                print(f"      ⚠️ Could not remove {file}: {e}")
        
        # Small delay to ensure filesystem sync
        time.sleep(0.5)
    
    def evaluate_all_models(self, specific_model: str = None) -> Dict[str, ModelResult]:
        """Evaluate all available models or a specific one."""
        # Clean up any existing cache before starting
        print("🧹 Initial cleanup...")
        self.cleanup_cache()
        
        if specific_model:
            models = [specific_model]
        else:
            models = self.get_available_models()
            print(f"🔍 Found {len(models)} models: {', '.join(models)}")
        
        results = {}
        
        for i, model in enumerate(models, 1):
            print(f"\n⏳ [{i}/{len(models)}] Evaluating {model}...")
            try:
                results[model] = self.test_model(model)
            except KeyboardInterrupt:
                print(f"\n⏹️ Evaluation cancelled by user")
                break
            except Exception as e:
                print(f"❌ Failed to test {model}: {e}")
                results[model] = ModelResult(
                    model_name=model,
                    total_questions=len(self.test_questions),
                    successful_responses=0,
                    total_time=0.0,
                    avg_response_time=0.0,
                    response_quality_scores=[],
                    responses={},
                    errors=[f"Model evaluation failed: {e}"]
                )
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, ModelResult]) -> str:
        """Generate comprehensive comparison report."""
        report = []
        report.append("🏆 OLLAMA MODEL EVALUATION RESULTS")
        report.append("=" * 70)
        
        # Sort models by combined score (success rate + quality)
        def calculate_combined_score(result: ModelResult) -> float:
            success_rate = result.successful_responses / result.total_questions if result.total_questions > 0 else 0
            avg_quality = sum(result.response_quality_scores) / len(result.response_quality_scores) if result.response_quality_scores else 0
            speed_bonus = max(0, (30 - result.avg_response_time) / 30 * 0.2)  # Speed bonus up to 0.2
            return (success_rate * 0.5) + (avg_quality * 0.4) + speed_bonus
        
        sorted_results = sorted(results.items(), key=lambda x: calculate_combined_score(x[1]), reverse=True)
        
        # Individual model results
        for i, (model_name, result) in enumerate(sorted_results, 1):
            success_rate = (result.successful_responses / result.total_questions * 100) if result.total_questions > 0 else 0
            avg_quality = sum(result.response_quality_scores) / len(result.response_quality_scores) if result.response_quality_scores else 0
            combined_score = calculate_combined_score(result)
            
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            report.append(f"\n{medal} {model_name}")
            report.append("-" * 50)
            report.append(f"  Success Rate: {success_rate:.1f}% ({result.successful_responses}/{result.total_questions})")
            report.append(f"  Avg Quality Score: {avg_quality:.2f}/1.00")
            report.append(f"  Avg Response Time: {result.avg_response_time:.1f}s")
            report.append(f"  Total Time: {result.total_time:.1f}s")
            report.append(f"  Combined Score: {combined_score:.3f}")
            
            # Performance category
            if result.avg_response_time < 10:
                perf = "🚀 Very Fast"
            elif result.avg_response_time < 20:
                perf = "⚡ Fast"
            elif result.avg_response_time < 30:
                perf = "🐢 Moderate"
            else:
                perf = "🐌 Slow"
            report.append(f"  Performance: {perf}")
            
            # Show sample responses for top model
            if i == 1 and result.responses:
                report.append(f"  📝 Sample Responses:")
                for question, response_data in list(result.responses.items())[:2]:
                    report.append(f"    Q: {question[:60]}...")
                    report.append(f"    A: {response_data['response'][:150]}...")
                    report.append(f"       (Quality: {response_data['quality_score']:.2f}, Time: {response_data['response_time']:.1f}s)")
            
            # Show errors if any
            if result.errors:
                report.append(f"  ⚠️ Errors: {len(result.errors)}")
                for error in result.errors[:2]:
                    report.append(f"    • {error[:80]}...")
        
        # Overall statistics
        report.append(f"\n📊 OVERALL STATISTICS")
        report.append("-" * 50)
        
        total_questions = sum(r.total_questions for r in results.values())
        total_successes = sum(r.successful_responses for r in results.values())
        avg_success_rate = (total_successes / total_questions * 100) if total_questions > 0 else 0
        
        report.append(f"  Models Evaluated: {len(results)}")
        report.append(f"  Total Questions: {total_questions}")
        report.append(f"  Total Successes: {total_successes}")
        report.append(f"  Overall Success Rate: {avg_success_rate:.1f}%")
        
        # Best in category
        if results:
            fastest_model = min(results.items(), key=lambda x: x[1].avg_response_time)
            highest_quality = max(results.items(), key=lambda x: sum(x[1].response_quality_scores) / len(x[1].response_quality_scores) if x[1].response_quality_scores else 0)
            most_reliable = max(results.items(), key=lambda x: x[1].successful_responses / x[1].total_questions if x[1].total_questions > 0 else 0)
            
            report.append(f"\n🏅 CATEGORY WINNERS")
            report.append("-" * 50)
            report.append(f"  🚀 Fastest: {fastest_model[0]} ({fastest_model[1].avg_response_time:.1f}s avg)")
            report.append(f"  💎 Highest Quality: {highest_quality[0]} ({sum(highest_quality[1].response_quality_scores) / len(highest_quality[1].response_quality_scores) if highest_quality[1].response_quality_scores else 0:.2f} avg)")
            report.append(f"  🎯 Most Reliable: {most_reliable[0]} ({most_reliable[1].successful_responses / most_reliable[1].total_questions * 100 if most_reliable[1].total_questions > 0 else 0:.1f}% success)")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")
        report.append("-" * 50)
        
        if sorted_results:
            best_model = sorted_results[0]
            report.append(f"  🌟 Recommended Model: {best_model[0]}")
            report.append(f"     Best overall performance with {calculate_combined_score(best_model[1]):.3f} combined score")
            
            if len(sorted_results) > 1:
                fast_alternative = min(sorted_results[:3], key=lambda x: x[1].avg_response_time)
                if fast_alternative[1].avg_response_time < best_model[1].avg_response_time:
                    report.append(f"  ⚡ Speed Alternative: {fast_alternative[0]}")
                    report.append(f"     Faster responses ({fast_alternative[1].avg_response_time:.1f}s vs {best_model[1].avg_response_time:.1f}s)")
        
        # Test environment
        report.append(f"\n📁 TEST ENVIRONMENT")
        report.append("-" * 50)
        report.append(f"  Repository: {self.repo_path}")
        report.append(f"  Questions per Model: {len(self.test_questions)}")
        report.append(f"  Quick Test Mode: {'Yes' if self.quick_test else 'No'}")
        
        return "\n".join(report)


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Ollama Model Evaluation Suite")
    parser.add_argument("--model", help="Test specific model only")
    parser.add_argument("--repo", default="/home/clutchcoder/working/rh_web/",
                       help="Repository path to test with")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer questions")
    parser.add_argument("--report", help="Save report to file")
    
    args = parser.parse_args()
    
    # Verify repository exists
    if not os.path.exists(args.repo):
        print(f"❌ Error: Repository not found at {args.repo}")
        return 1
    
    # Initialize evaluator
    evaluator = ModelEvaluator(repo_path=args.repo, quick_test=args.quick)
    
    try:
        # Run evaluation
        results = evaluator.evaluate_all_models(args.model)
        
        if not results:
            print("❌ No models were successfully evaluated")
            return 1
        
        # Generate and display report
        print("\n" + "=" * 70)
        print("📊 GENERATING COMPREHENSIVE MODEL COMPARISON...")
        print("=" * 70)
        
        report = evaluator.generate_comparison_report(results)
        print("\n" + report)
        
        # Save report
        if args.report:
            report_file = args.report
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"model_evaluation_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_file}")
        
        # Save detailed JSON results
        json_file = report_file.replace('.txt', '_detailed.json')
        json_data = {}
        for model_name, result in results.items():
            json_data[model_name] = {
                "model_name": result.model_name,
                "total_questions": result.total_questions,
                "successful_responses": result.successful_responses,
                "total_time": result.total_time,
                "avg_response_time": result.avg_response_time,
                "response_quality_scores": result.response_quality_scores,
                "avg_quality_score": sum(result.response_quality_scores) / len(result.response_quality_scores) if result.response_quality_scores else 0,
                "errors": result.errors,
                "responses": result.responses
            }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"📄 Detailed results saved to: {json_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n💥 Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())