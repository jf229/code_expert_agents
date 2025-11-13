"""Storage management for multi-project support.

Provides utilities for managing storage directories that are scoped to individual projects.
Each project gets its own subdirectory based on the repository name.
"""

import os
from pathlib import Path


def get_repo_name(repo_path: str) -> str:
    """Extract repository name from path (last directory name)."""
    return os.path.basename(os.path.normpath(repo_path))


def get_project_storage_dir(repo_path: str, base_storage_dir: str = "project_storage") -> str:
    """
    Get project-specific storage directory.

    Args:
        repo_path: Path to the repository being analyzed
        base_storage_dir: Base directory for all storage (default: "project_storage")

    Returns:
        Path to project-specific storage directory (created if doesn't exist)

    Example:
        repo_path = "/home/user/my_project"
        → storage dir = "project_storage/my_project/"
    """
    repo_name = get_repo_name(repo_path)
    project_storage = os.path.join(base_storage_dir, repo_name)

    # Create directory if it doesn't exist
    os.makedirs(project_storage, exist_ok=True)

    return project_storage


def get_project_file_path(repo_path: str, filename: str, base_storage_dir: str = "project_storage") -> str:
    """
    Get full path to a file in project-specific storage.

    Args:
        repo_path: Path to the repository being analyzed
        filename: Name of the file (e.g., "comprehensive_analysis.pkl")
        base_storage_dir: Base directory for all storage

    Returns:
        Full path to the file in project storage
    """
    project_dir = get_project_storage_dir(repo_path, base_storage_dir)
    return os.path.join(project_dir, filename)


def ensure_project_storage(repo_path: str, base_storage_dir: str = "project_storage") -> str:
    """Ensure project storage directory exists and return its path."""
    return get_project_storage_dir(repo_path, base_storage_dir)
