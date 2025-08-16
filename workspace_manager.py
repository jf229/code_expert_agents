#!/usr/bin/env python3
"""
Workspace Manager CLI

Command-line interface for managing multi-repository workspaces and 
intelligent repository configurations.

Usage Examples:
    # Analyze a single repository
    python workspace_manager.py analyze /path/to/repo
    
    # Create a multi-repository workspace
    python workspace_manager.py create-workspace my_project /path/repo1 /path/repo2
    
    # List all workspaces
    python workspace_manager.py list-workspaces
    
    # Generate intelligence report
    python workspace_manager.py report /path/to/repo
    
    # Get optimized config for a repository
    python workspace_manager.py config /path/to/repo
    
    # Auto-discover repositories in a directory
    python workspace_manager.py discover /path/to/projects
"""

import argparse
import sys
import yaml
import json
from pathlib import Path
from tabulate import tabulate

from repository_intelligence import RepositoryIntelligence, WorkspaceManager


def analyze_repository(repo_path: str):
    """Analyze a single repository and display results."""
    try:
        ri = RepositoryIntelligence()
        metadata = ri.analyze_repository(repo_path)
        
        print(f"\n🔍 Repository Analysis: {metadata.name}")
        print("=" * 50)
        
        # Basic info table
        basic_info = [
            ["Path", metadata.path],
            ["Type", metadata.repo_type],
            ["Size", f"{metadata.total_size_mb:.1f} MB"],
            ["Files", f"{metadata.file_count:,}"],
            ["Priority", metadata.processing_priority],
        ]
        
        if metadata.remote_url:
            basic_info.append(["Remote URL", metadata.remote_url])
        if metadata.current_branch:
            basic_info.append(["Branch", metadata.current_branch])
            
        print(tabulate(basic_info, headers=["Property", "Value"], tablefmt="grid"))
        
        # Language distribution
        if metadata.language_distribution:
            print("\n📊 Language Distribution:")
            lang_data = []
            total_files = sum(metadata.language_distribution.values())
            
            for lang, count in sorted(metadata.language_distribution.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / total_files) * 100
                lang_data.append([lang, count, f"{percentage:.1f}%"])
                
            print(tabulate(lang_data, headers=["Language", "Files", "Percentage"], tablefmt="grid"))
        
        # Project features
        features = []
        features.append(["Tests", "✅" if metadata.has_tests else "❌"])
        features.append(["Documentation", "✅" if metadata.has_docs else "❌"])
        features.append(["Configuration", "✅" if metadata.has_config else "❌"])
        
        if metadata.build_systems:
            features.append(["Build Systems", ", ".join(metadata.build_systems)])
        if metadata.frameworks:
            features.append(["Frameworks", ", ".join(metadata.frameworks)])
            
        print("\n🏗️ Project Features:")
        print(tabulate(features, headers=["Feature", "Status"], tablefmt="grid"))
        
        # Recommendations
        print(f"\n💡 Optimization Recommendations:")
        print(f"   • {len(metadata.recommended_excludes)} exclusion patterns recommended")
        print(f"   • Processing priority: {metadata.processing_priority}")
        if metadata.frameworks:
            print(f"   • Framework-specific optimizations available for: {', '.join(metadata.frameworks[:3])}")
            
    except Exception as e:
        print(f"❌ Error analyzing repository: {e}")
        return False
        
    return True


def create_workspace(name: str, repo_paths: list):
    """Create a new multi-repository workspace."""
    try:
        wm = WorkspaceManager()
        workspace = wm.create_workspace(name, repo_paths)
        
        print(f"\n✅ Created workspace: {workspace.name}")
        print(f"📁 Repositories ({len(workspace.repositories)}):")
        
        for i, repo in enumerate(workspace.repositories, 1):
            print(f"   {i}. {repo.name} ({repo.repo_type}) - {repo.total_size_mb:.1f} MB")
            if repo.primary_languages:
                print(f"      Languages: {', '.join(repo.primary_languages[:3])}")
                
        print(f"\n💾 Workspace saved to: .workspaces/{name}.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating workspace: {e}")
        return False


def list_workspaces():
    """List all available workspaces."""
    try:
        wm = WorkspaceManager()
        workspaces = wm.list_workspaces()
        
        if not workspaces:
            print("📭 No workspaces found.")
            return True
            
        print(f"\n📋 Available Workspaces ({len(workspaces)}):")
        print("=" * 40)
        
        workspace_data = []
        for ws_name in workspaces:
            workspace = wm.load_workspace(ws_name)
            if workspace:
                repo_count = len(workspace.repositories)
                created = workspace.created.strftime('%Y-%m-%d')
                workspace_data.append([ws_name, repo_count, created])
                
        print(tabulate(workspace_data, 
                      headers=["Workspace", "Repositories", "Created"], 
                      tablefmt="grid"))
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing workspaces: {e}")
        return False


def generate_report(repo_path: str, output_file: str = None):
    """Generate a comprehensive repository report."""
    try:
        ri = RepositoryIntelligence()
        report = ri.generate_report(repo_path)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to: {output_file}")
        else:
            print(report)
            
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False


def get_optimized_config(repo_path: str, output_file: str = None):
    """Get optimized configuration for a repository."""
    try:
        ri = RepositoryIntelligence()
        config = ri.get_optimized_config(repo_path)
        
        if output_file:
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"⚙️ Config saved to: {output_file}")
        else:
            print("⚙️ Optimized Configuration:")
            print(yaml.dump(config, default_flow_style=False))
            
        return True
        
    except Exception as e:
        print(f"❌ Error generating config: {e}")
        return False


def discover_repositories(base_path: str, create_workspace: bool = False):
    """Auto-discover repositories in a directory."""
    try:
        ri = RepositoryIntelligence()
        base = Path(base_path)
        
        if not base.exists():
            print(f"❌ Path does not exist: {base_path}")
            return False
            
        discovered_repos = []
        print(f"🔍 Scanning for repositories in: {base_path}")
        
        # Look for repositories
        for item in base.iterdir():
            if item.is_dir():
                try:
                    metadata = ri.analyze_repository(str(item))
                    discovered_repos.append((str(item), metadata))
                except ValueError:
                    # Not a repository
                    continue
                except Exception as e:
                    print(f"⚠️ Warning: Could not analyze {item}: {e}")
                    continue
                    
        if not discovered_repos:
            print("📭 No repositories found.")
            return True
            
        print(f"\n🎯 Discovered {len(discovered_repos)} repositories:")
        
        repo_data = []
        for repo_path, metadata in discovered_repos:
            languages = ", ".join(metadata.primary_languages[:2]) if metadata.primary_languages else "N/A"
            repo_data.append([
                metadata.name,
                metadata.repo_type,
                f"{metadata.total_size_mb:.1f} MB",
                languages
            ])
            
        print(tabulate(repo_data, 
                      headers=["Name", "Type", "Size", "Languages"], 
                      tablefmt="grid"))
        
        if create_workspace:
            workspace_name = f"discovered_{base.name}"
            repo_paths = [repo_path for repo_path, _ in discovered_repos]
            
            wm = WorkspaceManager()
            workspace = wm.create_workspace(workspace_name, repo_paths)
            print(f"\n✅ Created workspace: {workspace_name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error discovering repositories: {e}")
        return False


def load_workspace_config(workspace_name: str):
    """Load and activate a workspace configuration."""
    try:
        wm = WorkspaceManager()
        workspace = wm.load_workspace(workspace_name)
        
        if not workspace:
            print(f"❌ Workspace '{workspace_name}' not found.")
            return False
            
        print(f"📁 Loaded workspace: {workspace.name}")
        print(f"🏗️ Repositories ({len(workspace.repositories)}):")
        
        for i, repo in enumerate(workspace.repositories, 1):
            print(f"   {i}. {repo.name}")
            print(f"      Path: {repo.path}")
            print(f"      Languages: {', '.join(repo.primary_languages[:3])}")
            print()
            
        # Generate combined config
        combined_config = {
            'workspace': {
                'name': workspace.name,
                'primary_repo': workspace.primary_repo,
                'repositories': []
            }
        }
        
        for repo in workspace.repositories:
            repo_config = {
                'name': repo.name,
                'path': repo.path,
                'type': repo.repo_type,
                'languages': repo.primary_languages,
                'size_mb': repo.total_size_mb,
                'priority': repo.processing_priority
            }
            combined_config['workspace']['repositories'].append(repo_config)
            
        # Save combined config
        config_file = f"workspace_{workspace_name}_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(combined_config, f, default_flow_style=False)
            
        print(f"⚙️ Workspace config saved to: {config_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading workspace: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Workspace Manager - Intelligent Repository Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze /path/to/repo
  %(prog)s create-workspace my_project /repo1 /repo2
  %(prog)s list-workspaces
  %(prog)s report /path/to/repo --output report.md
  %(prog)s config /path/to/repo --output config.yaml
  %(prog)s discover /path/to/projects --create-workspace
  %(prog)s load-workspace my_project
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a repository')
    analyze_parser.add_argument('repo_path', help='Path to repository')
    
    # Create workspace command
    workspace_parser = subparsers.add_parser('create-workspace', help='Create a multi-repo workspace')
    workspace_parser.add_argument('name', help='Workspace name')
    workspace_parser.add_argument('repos', nargs='+', help='Repository paths')
    
    # List workspaces command
    subparsers.add_parser('list-workspaces', help='List all workspaces')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate repository report')
    report_parser.add_argument('repo_path', help='Path to repository')
    report_parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Generate optimized config')
    config_parser.add_argument('repo_path', help='Path to repository')
    config_parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Auto-discover repositories')
    discover_parser.add_argument('base_path', help='Base directory to scan')
    discover_parser.add_argument('--create-workspace', action='store_true',
                                help='Create workspace from discovered repos')
    
    # Load workspace command
    load_parser = subparsers.add_parser('load-workspace', help='Load and activate workspace')
    load_parser.add_argument('name', help='Workspace name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Execute commands
    success = False
    
    try:
        if args.command == 'analyze':
            success = analyze_repository(args.repo_path)
            
        elif args.command == 'create-workspace':
            success = create_workspace(args.name, args.repos)
            
        elif args.command == 'list-workspaces':
            success = list_workspaces()
            
        elif args.command == 'report':
            success = generate_report(args.repo_path, args.output)
            
        elif args.command == 'config':
            success = get_optimized_config(args.repo_path, args.output)
            
        elif args.command == 'discover':
            success = discover_repositories(args.base_path, args.create_workspace)
            
        elif args.command == 'load-workspace':
            success = load_workspace_config(args.name)
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())