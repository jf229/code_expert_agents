#!/usr/bin/env python3
"""
Repository Intelligence System

Provides intelligent repository detection, analysis, and management beyond static config.yaml variables.
Supports auto-detection, multi-repository workspaces, language-specific optimization, 
and dynamic repository insights.

Features:
- Auto-detect repository type, structure, and metadata
- Language-specific file filtering and processing
- Multi-repository workspace management
- Repository health analysis and optimization
- Dynamic ignore patterns and exclusions
- Repository caching and metadata persistence
"""

import os
import json
import pickle
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import yaml
from git import Repo, InvalidGitRepositoryError


@dataclass
class RepositoryMetadata:
    """Comprehensive repository metadata."""
    
    # Basic information
    name: str
    path: str
    repo_type: str  # git, svn, local
    
    # Git-specific metadata
    remote_url: Optional[str] = None
    current_branch: Optional[str] = None
    commit_hash: Optional[str] = None
    last_commit_date: Optional[datetime] = None
    
    # Language and structure analysis
    primary_languages: List[str] = field(default_factory=list)
    language_distribution: Dict[str, int] = field(default_factory=dict)
    file_count: int = 0
    total_size_mb: float = 0.0
    
    # Project structure insights
    has_tests: bool = False
    has_docs: bool = False
    has_config: bool = False
    build_systems: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    
    # Analysis metadata
    last_analyzed: datetime = field(default_factory=datetime.now)
    analysis_hash: str = ""
    
    # Performance optimization
    recommended_excludes: Set[str] = field(default_factory=set)
    processing_priority: str = "normal"  # low, normal, high


@dataclass
class WorkspaceConfig:
    """Multi-repository workspace configuration."""
    
    name: str
    repositories: List[RepositoryMetadata] = field(default_factory=list)
    primary_repo: Optional[str] = None
    shared_config: Dict = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)


class RepositoryDetector:
    """Intelligent repository detection and analysis."""
    
    # Language detection patterns
    LANGUAGE_PATTERNS = {
        'Python': ['.py', '.pyw', '.pyx'],
        'JavaScript': ['.js', '.jsx', '.mjs'],
        'TypeScript': ['.ts', '.tsx'],
        'Java': ['.java', '.class', '.jar'],
        'C++': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx'],
        'C': ['.c', '.h'],
        'C#': ['.cs', '.csx'],
        'Go': ['.go'],
        'Rust': ['.rs'],
        'Ruby': ['.rb', '.rbw'],
        'PHP': ['.php', '.phtml'],
        'Swift': ['.swift'],
        'Kotlin': ['.kt', '.kts'],
        'Scala': ['.scala', '.sc'],
        'R': ['.r', '.R'],
        'MATLAB': ['.m'],
        'Shell': ['.sh', '.bash', '.zsh'],
        'PowerShell': ['.ps1', '.psm1'],
        'HTML': ['.html', '.htm'],
        'CSS': ['.css', '.scss', '.sass', '.less'],
        'SQL': ['.sql'],
        'YAML': ['.yaml', '.yml'],
        'JSON': ['.json'],
        'XML': ['.xml', '.xsd', '.xsl'],
        'Markdown': ['.md', '.markdown'],
        'Dockerfile': ['Dockerfile', '.dockerfile'],
    }
    
    # Build system indicators
    BUILD_SYSTEMS = {
        'npm': ['package.json', 'package-lock.json', 'yarn.lock'],
        'pip': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
        'maven': ['pom.xml'],
        'gradle': ['build.gradle', 'build.gradle.kts', 'gradle.properties'],
        'cmake': ['CMakeLists.txt'],
        'make': ['Makefile', 'makefile'],
        'cargo': ['Cargo.toml', 'Cargo.lock'],
        'composer': ['composer.json', 'composer.lock'],
        'bundle': ['Gemfile', 'Gemfile.lock'],
        'sbt': ['build.sbt'],
        'bazel': ['BUILD', 'BUILD.bazel', 'WORKSPACE'],
    }
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        'React': ['react', 'jsx', 'tsx'],
        'Angular': ['@angular', 'angular.json'],
        'Vue': ['vue', '.vue'],
        'Django': ['django', 'manage.py'],
        'Flask': ['flask', 'app.py'],
        'FastAPI': ['fastapi', 'uvicorn'],
        'Spring': ['spring', '@SpringBootApplication'],
        'Express': ['express', 'app.js'],
        'Next.js': ['next', 'next.config'],
        'Nuxt': ['nuxt', 'nuxt.config'],
        'Rails': ['rails', 'Gemfile'],
        'Laravel': ['laravel', 'artisan'],
        'Unity': ['.unity', 'UnityEngine'],
        'Unreal': ['.uproject', 'UnrealEngine'],
    }
    
    # Standard exclusion patterns
    STANDARD_EXCLUDES = {
        'node_modules', '.git', '.svn', '.hg', '__pycache__', '.pytest_cache',
        'build', 'dist', 'target', 'bin', 'obj', '.vscode', '.idea',
        '.DS_Store', 'Thumbs.db', '*.log', '*.tmp', '.env', '.env.local',
        'coverage', '.coverage', '.nyc_output', 'logs', 'temp'
    }
    
    def __init__(self, cache_dir: str = ".repo_intelligence"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def detect_repository(self, path: str) -> RepositoryMetadata:
        """Detect and analyze repository at given path."""
        
        repo_path = Path(path).resolve()
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {path}")
            
        # Check cache first
        cache_key = self._get_cache_key(str(repo_path))
        cached_metadata = self._load_cached_metadata(cache_key)
        
        if cached_metadata and self._is_cache_valid(cached_metadata, repo_path):
            return cached_metadata
            
        print(f"Analyzing repository: {repo_path}")
        
        # Create new metadata
        metadata = RepositoryMetadata(
            name=repo_path.name,
            path=str(repo_path)
        )
        
        # Detect repository type
        metadata.repo_type = self._detect_repo_type(repo_path)
        
        # Get Git information if applicable
        if metadata.repo_type == "git":
            self._analyze_git_repo(repo_path, metadata)
            
        # Analyze file structure and languages
        self._analyze_file_structure(repo_path, metadata)
        
        # Detect build systems and frameworks
        self._detect_build_systems(repo_path, metadata)
        self._detect_frameworks(repo_path, metadata)
        
        # Generate recommendations
        self._generate_recommendations(metadata)
        
        # Cache the results
        self._cache_metadata(cache_key, metadata)
        
        return metadata
        
    def _detect_repo_type(self, repo_path: Path) -> str:
        """Detect repository type (git, svn, local)."""
        try:
            Repo(repo_path)
            return "git"
        except InvalidGitRepositoryError:
            pass
            
        if (repo_path / ".svn").exists():
            return "svn"
            
        return "local"
        
    def _analyze_git_repo(self, repo_path: Path, metadata: RepositoryMetadata):
        """Analyze Git repository metadata."""
        try:
            repo = Repo(repo_path)
            
            # Basic Git info
            metadata.current_branch = repo.active_branch.name
            metadata.commit_hash = repo.head.commit.hexsha
            metadata.last_commit_date = datetime.fromtimestamp(repo.head.commit.committed_date)
            
            # Remote URL
            if repo.remotes:
                metadata.remote_url = repo.remotes.origin.url
                
        except Exception as e:
            print(f"Warning: Could not analyze Git repository: {e}")
            
    def _analyze_file_structure(self, repo_path: Path, metadata: RepositoryMetadata):
        """Analyze file structure and language distribution."""
        
        language_counts = {}
        total_files = 0
        total_size = 0
        
        # Walk through directory
        for root, dirs, files in os.walk(repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]
            
            root_path = Path(root)
            
            for file in files:
                if self._should_exclude_file(file):
                    continue
                    
                file_path = root_path / file
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    total_files += 1
                    
                    # Detect language
                    language = self._detect_file_language(file_path)
                    if language:
                        language_counts[language] = language_counts.get(language, 0) + 1
                        
                except (OSError, PermissionError):
                    continue
                    
        # Set metadata
        metadata.file_count = total_files
        metadata.total_size_mb = total_size / (1024 * 1024)
        metadata.language_distribution = language_counts
        
        # Determine primary languages (top 3)
        sorted_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        metadata.primary_languages = [lang for lang, count in sorted_languages[:3]]
        
        # Check for common directory patterns
        metadata.has_tests = self._has_test_directory(repo_path)
        metadata.has_docs = self._has_docs_directory(repo_path)
        metadata.has_config = self._has_config_files(repo_path)
        
    def _detect_file_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        name = file_path.name
        
        for language, extensions in self.LANGUAGE_PATTERNS.items():
            if suffix in extensions or name in extensions:
                return language
                
        return None
        
    def _detect_build_systems(self, repo_path: Path, metadata: RepositoryMetadata):
        """Detect build systems used in the repository."""
        
        detected_systems = []
        
        for system, indicators in self.BUILD_SYSTEMS.items():
            for indicator in indicators:
                if (repo_path / indicator).exists():
                    detected_systems.append(system)
                    break
                    
        metadata.build_systems = detected_systems
        
    def _detect_frameworks(self, repo_path: Path, metadata: RepositoryMetadata):
        """Detect frameworks used in the repository."""
        
        detected_frameworks = []
        
        # Check package.json for JS frameworks
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    package_data = json.load(f)
                    deps = {**package_data.get('dependencies', {}), 
                           **package_data.get('devDependencies', {})}
                    
                    for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                        for pattern in patterns:
                            if any(pattern in dep for dep in deps.keys()):
                                detected_frameworks.append(framework)
                                break
                                
            except (json.JSONDecodeError, FileNotFoundError):
                pass
                
        # Check requirements.txt for Python frameworks
        req_files = ['requirements.txt', 'pyproject.toml', 'setup.py']
        for req_file in req_files:
            req_path = repo_path / req_file
            if req_path.exists():
                try:
                    content = req_path.read_text().lower()
                    for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                        for pattern in patterns:
                            if pattern.lower() in content:
                                detected_frameworks.append(framework)
                                break
                except FileNotFoundError:
                    continue
                    
        metadata.frameworks = list(set(detected_frameworks))
        
    def _generate_recommendations(self, metadata: RepositoryMetadata):
        """Generate optimization recommendations."""
        
        excludes = set(self.STANDARD_EXCLUDES)
        
        # Language-specific excludes
        if 'Python' in metadata.primary_languages:
            excludes.update({'*.pyc', '__pycache__', '.pytest_cache', 'venv', '.venv'})
            
        if 'JavaScript' in metadata.primary_languages or 'TypeScript' in metadata.primary_languages:
            excludes.update({'node_modules', 'bower_components', '.next', 'dist'})
            
        if 'Java' in metadata.primary_languages:
            excludes.update({'target', '*.class', '.gradle'})
            
        if 'C++' in metadata.primary_languages or 'C' in metadata.primary_languages:
            excludes.update({'build', 'cmake-build-*', '*.o', '*.exe'})
            
        # Build system specific excludes
        if 'npm' in metadata.build_systems:
            excludes.update({'node_modules', 'npm-debug.log'})
            
        if 'maven' in metadata.build_systems or 'gradle' in metadata.build_systems:
            excludes.update({'target', 'build'})
            
        metadata.recommended_excludes = excludes
        
        # Set processing priority based on size
        if metadata.total_size_mb > 1000:  # > 1GB
            metadata.processing_priority = "low"
        elif metadata.total_size_mb > 100:  # > 100MB
            metadata.processing_priority = "normal"
        else:
            metadata.processing_priority = "high"
            
    def _should_exclude_dir(self, dirname: str) -> bool:
        """Check if directory should be excluded."""
        return dirname in self.STANDARD_EXCLUDES or dirname.startswith('.')
        
    def _should_exclude_file(self, filename: str) -> bool:
        """Check if file should be excluded."""
        if filename.startswith('.'):
            return True
        if filename.endswith(('.log', '.tmp', '.bak', '.swp')):
            return True
        return False
        
    def _has_test_directory(self, repo_path: Path) -> bool:
        """Check if repository has test directories."""
        test_indicators = ['test', 'tests', 'spec', 'specs', '__tests__']
        for indicator in test_indicators:
            if (repo_path / indicator).exists():
                return True
        return False
        
    def _has_docs_directory(self, repo_path: Path) -> bool:
        """Check if repository has documentation."""
        doc_indicators = ['docs', 'doc', 'documentation', 'README.md', 'README.rst']
        for indicator in doc_indicators:
            if (repo_path / indicator).exists():
                return True
        return False
        
    def _has_config_files(self, repo_path: Path) -> bool:
        """Check if repository has configuration files."""
        config_indicators = ['.env.example', 'config.yaml', 'config.json', 'settings.py']
        for indicator in config_indicators:
            if (repo_path / indicator).exists():
                return True
        return False
        
    def _get_cache_key(self, repo_path: str) -> str:
        """Generate cache key for repository."""
        return hashlib.md5(repo_path.encode()).hexdigest()
        
    def _load_cached_metadata(self, cache_key: str) -> Optional[RepositoryMetadata]:
        """Load cached repository metadata."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None
        
    def _cache_metadata(self, cache_key: str, metadata: RepositoryMetadata):
        """Cache repository metadata."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            print(f"Warning: Could not cache metadata: {e}")
            
    def _is_cache_valid(self, metadata: RepositoryMetadata, repo_path: Path) -> bool:
        """Check if cached metadata is still valid."""
        
        # Cache expires after 24 hours
        if (datetime.now() - metadata.last_analyzed).days > 1:
            return False
            
        # Check if repository has been modified
        try:
            if metadata.repo_type == "git":
                repo = Repo(repo_path)
                if repo.head.commit.hexsha != metadata.commit_hash:
                    return False
        except Exception:
            pass
            
        return True


class WorkspaceManager:
    """Manages multi-repository workspaces."""
    
    def __init__(self, workspace_dir: str = ".workspaces"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        self.detector = RepositoryDetector()
        
    def create_workspace(self, name: str, repo_paths: List[str]) -> WorkspaceConfig:
        """Create a new multi-repository workspace."""
        
        repositories = []
        for repo_path in repo_paths:
            metadata = self.detector.detect_repository(repo_path)
            repositories.append(metadata)
            
        workspace = WorkspaceConfig(
            name=name,
            repositories=repositories,
            primary_repo=repositories[0].path if repositories else None
        )
        
        self._save_workspace(workspace)
        return workspace
        
    def load_workspace(self, name: str) -> Optional[WorkspaceConfig]:
        """Load an existing workspace."""
        workspace_file = self.workspace_dir / f"{name}.pkl"
        try:
            if workspace_file.exists():
                with open(workspace_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None
        
    def list_workspaces(self) -> List[str]:
        """List all available workspaces."""
        return [f.stem for f in self.workspace_dir.glob("*.pkl")]
        
    def _save_workspace(self, workspace: WorkspaceConfig):
        """Save workspace configuration."""
        workspace_file = self.workspace_dir / f"{workspace.name}.pkl"
        with open(workspace_file, 'wb') as f:
            pickle.dump(workspace, f)


class RepositoryIntelligence:
    """Main interface for intelligent repository management."""
    
    def __init__(self):
        self.detector = RepositoryDetector()
        self.workspace_manager = WorkspaceManager()
        
    def analyze_repository(self, path: str) -> RepositoryMetadata:
        """Analyze a single repository."""
        return self.detector.detect_repository(path)
        
    def get_optimized_config(self, repo_path: str) -> Dict:
        """Get optimized configuration for a repository."""
        
        metadata = self.detector.detect_repository(repo_path)
        
        config = {
            'repository': {
                'local_path': metadata.path,
                'remote_url': metadata.remote_url,
                'name': metadata.name,
                'type': metadata.repo_type,
                'primary_languages': metadata.primary_languages,
                'processing_priority': metadata.processing_priority
            },
            'intelligent_filtering': {
                'exclude_patterns': list(metadata.recommended_excludes),
                'language_focus': metadata.primary_languages[:2],  # Top 2 languages
                'framework_aware': metadata.frameworks
            },
            'optimization': {
                'file_count': metadata.file_count,
                'size_mb': metadata.total_size_mb,
                'recommended_batch_size': self._calculate_batch_size(metadata),
                'parallel_processing': metadata.processing_priority == "high"
            }
        }
        
        return config
        
    def create_smart_workspace(self, name: str, base_paths: List[str]) -> WorkspaceConfig:
        """Create an intelligent workspace with auto-detected repositories."""
        
        detected_repos = []
        
        for base_path in base_paths:
            base = Path(base_path)
            if base.is_dir():
                # Look for repositories in subdirectories
                for item in base.iterdir():
                    if item.is_dir():
                        try:
                            metadata = self.detector.detect_repository(str(item))
                            detected_repos.append(str(item))
                        except ValueError:
                            continue
            else:
                # Single repository
                try:
                    self.detector.detect_repository(base_path)
                    detected_repos.append(base_path)
                except ValueError:
                    continue
                    
        return self.workspace_manager.create_workspace(name, detected_repos)
        
    def generate_report(self, repo_path: str) -> str:
        """Generate a comprehensive repository analysis report."""
        
        metadata = self.detector.detect_repository(repo_path)
        
        report = f"""
# Repository Intelligence Report: {metadata.name}

## Basic Information
- **Path**: {metadata.path}
- **Type**: {metadata.repo_type}
- **Size**: {metadata.total_size_mb:.1f} MB
- **Files**: {metadata.file_count:,}
- **Last Analyzed**: {metadata.last_analyzed.strftime('%Y-%m-%d %H:%M:%S')}

## Git Information
- **Remote URL**: {metadata.remote_url or 'N/A'}
- **Current Branch**: {metadata.current_branch or 'N/A'}
- **Last Commit**: {metadata.last_commit_date.strftime('%Y-%m-%d %H:%M:%S') if metadata.last_commit_date else 'N/A'}

## Language Distribution
"""
        
        for lang, count in sorted(metadata.language_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / metadata.file_count) * 100
            report += f"- **{lang}**: {count} files ({percentage:.1f}%)\n"
            
        report += f"""
## Project Structure
- **Has Tests**: {'✅' if metadata.has_tests else '❌'}
- **Has Documentation**: {'✅' if metadata.has_docs else '❌'}
- **Has Configuration**: {'✅' if metadata.has_config else '❌'}

## Build Systems & Frameworks
- **Build Systems**: {', '.join(metadata.build_systems) or 'None detected'}
- **Frameworks**: {', '.join(metadata.frameworks) or 'None detected'}

## Optimization Recommendations
- **Processing Priority**: {metadata.processing_priority}
- **Recommended Excludes**: {len(metadata.recommended_excludes)} patterns
- **Suggested Batch Size**: {self._calculate_batch_size(metadata)}

## Recommended Config Enhancement
```yaml
repository:
  local_path: "{metadata.path}"
  intelligent_filtering:
    primary_languages: {metadata.primary_languages}
    exclude_patterns: {list(metadata.recommended_excludes)[:5]}  # Top 5 shown
    processing_priority: {metadata.processing_priority}
```
"""
        
        return report
        
    def _calculate_batch_size(self, metadata: RepositoryMetadata) -> int:
        """Calculate optimal batch size for processing."""
        
        if metadata.total_size_mb > 1000:  # > 1GB
            return 10
        elif metadata.total_size_mb > 100:  # > 100MB
            return 25
        else:
            return 50


# CLI Interface
def main():
    """CLI for repository intelligence."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Intelligence System")
    parser.add_argument("command", choices=['analyze', 'workspace', 'report', 'config'])
    parser.add_argument("--path", help="Repository path")
    parser.add_argument("--name", help="Workspace name")
    parser.add_argument("--repos", nargs="+", help="Repository paths for workspace")
    
    args = parser.parse_args()
    
    ri = RepositoryIntelligence()
    
    if args.command == "analyze":
        if not args.path:
            print("Error: --path required for analyze command")
            return
            
        metadata = ri.analyze_repository(args.path)
        print(f"Repository: {metadata.name}")
        print(f"Type: {metadata.repo_type}")
        print(f"Languages: {', '.join(metadata.primary_languages)}")
        print(f"Size: {metadata.total_size_mb:.1f} MB")
        
    elif args.command == "workspace":
        if not args.name or not args.repos:
            print("Error: --name and --repos required for workspace command")
            return
            
        workspace = ri.create_smart_workspace(args.name, args.repos)
        print(f"Created workspace '{workspace.name}' with {len(workspace.repositories)} repositories")
        
    elif args.command == "report":
        if not args.path:
            print("Error: --path required for report command")
            return
            
        report = ri.generate_report(args.path)
        print(report)
        
    elif args.command == "config":
        if not args.path:
            print("Error: --path required for config command")
            return
            
        config = ri.get_optimized_config(args.path)
        print(yaml.dump(config, default_flow_style=False))


if __name__ == "__main__":
    main()