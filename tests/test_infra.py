"""
tests/test_infra.py: Infrastructure, Dependency, Docker & CLI Verification Tests.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import pytest
import yaml

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_reproduction import parse_args, run_reproduction
from scripts.setup_dev import (
    check_dependencies,
    check_python_version,
    create_directories,
    setup_env_file,
)


class TestRequirementsAndPackaging:
    """Verify package dependency declarations and PEP 517/621 packaging."""

    def test_requirements_file_exists_and_valid(self):
        req_path = ROOT_DIR / "requirements.txt"
        assert req_path.exists(), "requirements.txt must exist at repository root"

        with open(req_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        assert len(lines) > 0, "requirements.txt must contain dependency declarations"

        # Check required packages
        packages = [line.split(">=")[0].split("==")[0].split("<")[0].strip() for line in lines]
        expected_pkgs = {
            "numpy",
            "scipy",
            "pandas",
            "scikit-learn",
            "pydantic",
            "pydantic-settings",
            "matplotlib",
            "seaborn",
            "openai",
            "pytest",
        }
        for pkg in expected_pkgs:
            assert pkg in packages, f"Expected package '{pkg}' in requirements.txt"

    def test_pyproject_toml_validity(self):
        pyproject_path = ROOT_DIR / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml must exist at repository root"

        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse TOML
        try:
            import tomllib
            data = tomllib.loads(content)
        except ImportError:
            import tomli
            data = tomli.loads(content)

        assert "project" in data, "pyproject.toml missing [project] table"
        assert data["project"]["name"] == "agent-memory-manage"
        assert data["project"]["requires-python"] == ">=3.10"
        assert "dependencies" in data["project"]
        assert len(data["project"]["dependencies"]) >= 10
        assert "project.scripts" in data or "scripts" in data.get("project", {})
        assert "tool" in data and "pytest" in data["tool"]
        assert "ruff" in data["tool"]


class TestDockerAndCompose:
    """Verify multi-stage Dockerfile and Docker Compose infrastructure."""

    def test_dockerfile_multi_stage_structure(self):
        dockerfile_path = ROOT_DIR / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile must exist at repository root"

        with open(dockerfile_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check multi-stage build structure
        assert "FROM python:3.10-slim AS builder" in content
        assert "FROM python:3.10-slim AS runner" in content

        # Check non-root security principles
        assert "useradd" in content and "appuser" in content
        assert "USER appuser" in content

        # Check workdir and entrypoint
        assert "WORKDIR /app" in content
        assert 'ENTRYPOINT ["python", "scripts/run_reproduction.py"]' in content

    def test_docker_compose_validity(self):
        compose_path = ROOT_DIR / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml must exist at repository root"

        with open(compose_path, "r", encoding="utf-8") as f:
            compose_data = yaml.safe_load(f)

        assert "services" in compose_data, "docker-compose.yml must define services"
        services = compose_data["services"]

        expected_services = ["test", "reproduction", "protocol-a", "protocol-b", "protocol-c", "protocol-d"]
        for svc in expected_services:
            assert svc in services, f"Service '{svc}' missing in docker-compose.yml"
            assert "build" in services[svc], f"Service '{svc}' missing build section"


class TestEnvironmentAndCI:
    """Verify environment configuration templates and GitHub Actions CI."""

    def test_env_example_completeness(self):
        env_example_path = ROOT_DIR / ".env.example"
        assert env_example_path.exists(), ".env.example must exist at repository root"

        with open(env_example_path, "r", encoding="utf-8") as f:
            content = f.read()

        required_vars = [
            "OPENAI_API_KEY",
            "MODEL_BACKBONE",
            "LOG_LEVEL",
            "RESULTS_DIR",
            "RANDOM_SEED",
            "BENCHMARK_ENV",
            "NUM_STEPS",
            "INITIAL_MEMORY_SIZE",
        ]
        for var in required_vars:
            assert f"{var}=" in content, f"Variable '{var}' missing in .env.example"

    def test_github_ci_workflow_validity(self):
        ci_path = ROOT_DIR / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists(), ".github/workflows/ci.yml must exist"

        with open(ci_path, "r", encoding="utf-8") as f:
            ci_data = yaml.safe_load(f)

        assert "name" in ci_data
        assert "on" in ci_data or True in ci_data, "ci.yml missing trigger events"
        assert "jobs" in ci_data
        assert "lint-and-test" in ci_data["jobs"]
        assert "docker-build" in ci_data["jobs"]


class TestCLIRunnerAndScripts:
    """Verify standalone reproduction CLI and setup scripts."""

    def test_parse_args_defaults(self):
        args = parse_args([])
        assert args.protocol == "A"
        assert args.env == "reg_agent"
        assert args.steps == 1000
        assert args.init_mem_size == 100
        assert args.capacity == 100
        assert args.seed == 42
        assert args.output_dir == "./results"
        assert args.dry_run is False

    @pytest.mark.parametrize("protocol", ["A", "B", "C", "D", "all"])
    def test_parse_args_all_protocols(self, protocol):
        args = parse_args(["--protocol", protocol, "--steps", "500", "--dry-run"])
        assert args.protocol == protocol
        assert args.steps == 500
        assert args.dry_run is True

    def test_parse_args_invalid_protocol(self):
        with pytest.raises(SystemExit):
            parse_args(["--protocol", "INVALID"])

    def test_run_reproduction_dry_run_all_protocols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args([
                "--protocol", "all",
                "--output-dir", tmpdir,
                "--steps", "200",
                "--dry-run",
            ])
            exit_code = run_reproduction(args)
            assert exit_code == 0

            # Verify output files generated
            manifest_file = Path(tmpdir) / "reproduction_manifest.json"
            assert manifest_file.exists()

            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["status"] == "success"
            assert "A" in manifest["protocols_executed"]
            assert "B" in manifest["protocols_executed"]
            assert "C" in manifest["protocols_executed"]
            assert "D" in manifest["protocols_executed"]

            for proto in ["a", "b", "c", "d"]:
                proto_file = Path(tmpdir) / f"protocol_{proto}_result.json"
                assert proto_file.exists()

    def test_setup_dev_script_functions(self):
        assert check_python_version() is True
        assert check_dependencies() is True

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test create directories
            create_directories()


class TestDocumentationIntegrity:
    """Verify comprehensive documentation exists and contains essential sections."""

    def test_documentation_files_exist(self):
        dev_doc = ROOT_DIR / "docs" / "DEVELOPMENT.md"
        repro_doc = ROOT_DIR / "docs" / "REPRODUCIBILITY.md"

        assert dev_doc.exists(), "docs/DEVELOPMENT.md must exist"
        assert repro_doc.exists(), "docs/REPRODUCIBILITY.md must exist"

        with open(dev_doc, "r", encoding="utf-8") as f:
            dev_content = f.read()
        with open(repro_doc, "r", encoding="utf-8") as f:
            repro_content = f.read()

        assert "Prerequisites" in dev_content or "System Requirements" in dev_content
        assert "Docker" in dev_content
        assert "Protocol A" in repro_content
        assert "Protocol B" in repro_content
        assert "Protocol C" in repro_content
        assert "Protocol D" in repro_content
