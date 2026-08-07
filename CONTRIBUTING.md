# Contributing to RCPSP-CF-IVFTH

👍 Thanks for your interest in improving this project! This document explains how to set up your development environment, propose changes, and help with reviews.

## Table of Contents
- [Ways to Contribute](#ways-to-contribute)
- [Development Workflow](#development-workflow)
- [Environment Setup](#environment-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)
- [Community Expectations](#community-expectations)

## Ways to Contribute
- **Report bugs** by opening an issue using the bug-report template.
- **Suggest enhancements** through the feature-request template.
- **Submit pull requests** that fix bugs, add features, improve docs, or refactor code.
- **Triage issues** by confirming reproducible bugs, labelling, or suggesting workarounds.
- **Improve documentation** in `README.md`, `docs/`, or docstrings.
- **Add tests** that increase coverage or prevent regressions.

## Development Workflow
1. Fork the repository and create a topic branch from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/short-description
   ```
2. Install dependencies and set up tooling (see below).
3. Make your changes, keeping commits focused and descriptive.
4. Run the full test suite and lint checks locally.
5. Update documentation, changelog entries, or config files if needed.
6. Push your branch and open a pull request referencing any related issues.
7. Participate in code review, adjust as requested, and squash commits if asked.

## Environment Setup
RCPSP-CF-IVFTH is a Python package that relies on Pyomo and an MILP solver.

1. **Install Python 3.10+** (3.11/3.12 are tested).
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -U pip wheel
   pip install -e .[dev]
   ```
   The `dev` extras include pytest, coverage, linters, and documentation tooling.
4. **Install a solver** compatible with Pyomo, e.g.:
   - CBC via conda: `conda install -c conda-forge coincbc`
   - GLPK (Linux/macOS): package manager install
   - Commercial solvers (Gurobi, CPLEX) if licensed.
5. **Verify installation**:
   ```bash
   python -m pytest
   ```

> ℹ️ Solver availability is optional for unit tests; tests skip solver runs if none is found.

## Coding Standards
- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python style. Run `ruff format` and `ruff check` before committing.
- Type annotations are encouraged. Use `mypy` for static checking (`mypy rcpsp_cf_ivfth`).
- Keep functions focused and add docstrings explaining business logic.
- Avoid introducing new dependencies unless they are essential.
- When editing scientific content, cite the source or explain the reasoning in comments.

## Testing
- Unit tests live in `tests/` and use `pytest`.
- Run `python -m pytest` before submitting a PR.
- Add regression tests when fixing bugs or introducing new features.
- Use parametrized tests to cover multiple scenarios.
- For solver-dependent tests, rely on fixtures that skip when a solver is unavailable.

## Documentation
- Update `docs/` Markdown files when workflows or APIs change.
- Regenerate the API docs after changing docstrings:
  ```bash
  python docs/_generate_api_md.py
  ```
- Keep the README concise; link to deeper documentation when necessary.
- Include usage examples for new public APIs.

## Submitting Changes
- Use meaningful commit messages (imperative mood): `Add IVF-TH weight validation`.
- Open a pull request against `main` (unless instructed otherwise).
- Ensure the PR description includes:
  - Summary of the change.
  - Linked issue numbers (e.g. `Closes #42`).
  - Testing performed (`pytest`, `ruff`, etc.).
  - Backwards-compatibility considerations.
- PRs require at least one maintainer approval before merge.

## Release Process
Releases are cut by hand, deliberately: publishing a release mints a Zenodo DOI, which
cannot be withdrawn afterwards, so it should never happen as a side effect of a merge.

1. Ensure `main` is green and up to date.
2. Update the version in `pyproject.toml` (the package reads it from installed metadata
   at runtime, so this is the single source of truth).
3. Update the release dates in `CITATION.cff` (both the top level and the
   `preferred-citation` block) and `datePublished` in `codemeta.json`. Leave
   `dateCreated` alone.
4. Add the version heading and its notes to `CHANGELOG.md`.
5. Merge the above, then tag the merge commit on `main` and push the tag:
   ```bash
   git checkout main && git pull
   git tag -a v2.1.0 -m "v2.1.0"
   git push origin v2.1.0
   ```
6. Publish the release, writing notes that summarise the change and any migration steps:
   ```bash
   gh release create v2.1.0 --title v2.1.0 --notes-file RELEASE_NOTES.md
   ```
   If the release changes what the model computes, say so at the top of the notes:
   anyone citing an earlier DOI needs to know their results are not comparable.
7. Confirm the DOI appears on Zenodo and that the badge in `README.md` still resolves.

## Community Expectations
- Be kind and constructive in all communications.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Respect review cycles and give reviewers enough time to respond.
- Assume good intent and provide actionable feedback.
- If you see something, say something—help keep the community safe and welcoming.

We appreciate your effort in making RCPSP-CF-IVFTH better! If you have questions, open an issue or reach out to the maintainers. Happy contributing!
