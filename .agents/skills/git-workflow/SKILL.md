---
name: git-workflow
description: >-
  Standardized Git version control workflow, conventional commits, branch management,
  pull request creation, atomic commit discipline, and clean history maintenance.
  Use when committing changes, creating branches, opening PRs, or resolving Git conflicts.
---

# Git Workflow & Version Control Standards

Disciplined Git workflow for reproducible research and production codebases.

## When to Activate

- Creating new feature, bugfix, or experiment branches
- Writing commit messages (conventional commit compliance)
- Opening Pull Requests (PRs) with structured test plans
- Performing rebases, squashes, or history cleanups
- Reviewing Git diffs and branch integrity

---

## 1. Conventional Commit Standard

Format every commit message strictly according to the Conventional Commits specification:

```text
<type>(<scope>): <short imperative description>

[optional body with rationale and technical details]

[optional footer, e.g., Closes #123, BREAKING CHANGE: ...]
```

### Allowed Types
| Type | Purpose | Example |
| :--- | :--- | :--- |
| `feat` | New feature or capability | `feat(memory): add hybrid vector-keyword retrieval` |
| `fix` | Bug fix in existing logic | `fix(agent): handle token limit overflow in context window` |
| `refactor` | Code restructuring without behavior change | `refactor(eval): decouple metric computation from runner` |
| `test` | Adding or updating tests | `test(memory): add property-based tests for lru cache` |
| `perf` | Performance improvement | `perf(retrieval): index embeddings with HNSW cosine index` |
| `docs` | Documentation only | `docs(readme): add benchmark reproduction instructions` |
| `chore` | Build tasks, dependencies, tooling | `chore(deps): bump pydantic to 2.8.0` |
| `ci` | CI/CD pipeline changes | `ci(github): add matrix testing for python 3.10 and 3.11` |

### Rules
- Keep the first line (subject) under 72 characters.
- Use imperative mood: "add" not "added", "fix" not "fixes".
- No ending period on the subject line.
- Separate subject from body with a blank line.

---

## 2. Branching Strategy

```text
main (stable, production-grade)
  ├── feature/<feature-name>       # e.g., feature/hierarchical-memory
  ├── exp/<experiment-name>        # e.g., exp/acl2026-baseline-eval
  ├── fix/<bug-description>        # e.g., fix/memory-leak-async-loop
  └── chore/<maintenance-task>     # e.g., chore/update-docker-base
```

---

## 3. Pull Request Guidelines

Before opening a PR:
1. Ensure all tests pass: `pytest`
2. Run linters & type checks: `ruff check .` and `mypy .`
3. Inspect the full branch diff: `git diff main...HEAD`
4. Write a structured PR description using the following template:

```markdown
## Summary
- Detailed bullet points summarizing what changed.

## Motivation & Architecture
- Why this change was made and how it fits the system design.

## Test Plan & Verification
- [x] Unit tests passed (`pytest tests/unit/`)
- [x] Integration tests passed (`pytest tests/integration/`)
- [x] Benchmark verification (`python scripts/run_benchmark.py`)

## Checklist
- [x] Type hints verified with mypy
- [x] No secrets or temporary artifacts committed
- [x] Documentation updated
```

---

## 4. Atomic Commits & Clean History

- **Atomic Commits**: Each commit must represent a single logical change that leaves the test suite passing.
- **No W.I.P. Commits on Main**: Use `git rebase -i` to squash intermediary scratch commits before merging.
- **Never force-push to `main`**.
