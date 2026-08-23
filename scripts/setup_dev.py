#!/usr/bin/env python3
"""
scripts/setup_dev.py: Developer Environment Initialization & Verification Script

Ensures local Python runtime compatibility (Python >= 3.10), checks critical
dependencies, initializes template environment configurations, and creates
workspace directory trees.
"""

import importlib
import os
import shutil
import sys
from pathlib import Path

REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 10

CRITICAL_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "pydantic",
    "openai",
    "matplotlib",
    "seaborn",
    "tqdm",
    "pytest",
]


def check_python_version() -> bool:
    """Verify current Python version meets >= 3.10 requirement."""
    print("Checking Python version...")
    cur = sys.version_info
    if (cur.major, cur.minor) < (REQUIRED_PYTHON_MAJOR, REQUIRED_PYTHON_MINOR):
        print(
            f"[ERROR] Python {REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}+ required. "
            f"Detected Python {cur.major}.{cur.minor}.{cur.micro}."
        )
        return False
    print(f"[OK] Python version: {cur.major}.{cur.minor}.{cur.micro}")
    return True


def check_dependencies() -> bool:
    """Check availability of core scientific and agent dependencies."""
    print("\nChecking core Python dependencies...")
    all_ok = True
    for pkg in CRITICAL_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pkg:<15} (version: {ver})")
        except ImportError:
            print(f"  [MISSING] {pkg:<15} (run: pip install -r requirements.txt)")
            all_ok = False
    return all_ok


def setup_env_file() -> None:
    """Initialize .env from .env.example if not present."""
    print("\nChecking .env configuration...")
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    example_file = root / ".env.example"

    if not env_file.exists():
        if example_file.exists():
            shutil.copyfile(example_file, env_file)
            print(f"  [CREATED] Created '{env_file.name}' from '{example_file.name}'.")
        else:
            print(f"  [WARNING] '{example_file.name}' not found.")
    else:
        print(f"  [OK] '{env_file.name}' already exists.")


def create_directories() -> None:
    """Ensure required runtime and artifact directories exist."""
    print("\nEnsuring runtime directories...")
    root = Path(__file__).resolve().parent.parent
    dirs = [
        root / "results",
        root / "docs",
        root / "tests",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Directory verified: {d.relative_to(root)}")


def main() -> int:
    """Run all verification steps."""
    print("=" * 60)
    print("Agent Memory Management - Developer Setup & Verification")
    print("=" * 60)

    py_ok = check_python_version()
    dep_ok = check_dependencies()
    setup_env_file()
    create_directories()

    print("\n" + "=" * 60)
    if py_ok and dep_ok:
        print("[SUCCESS] Environment is fully configured and ready for experiments!")
        print("Run benchmarks via: python scripts/run_reproduction.py --help")
        print("Run test suite via: pytest tests/")
        print("=" * 60)
        return 0
    else:
        print("[ACTION REQUIRED] Please install missing dependencies before proceeding:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
