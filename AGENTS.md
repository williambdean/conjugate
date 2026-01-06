# Agent Guidelines for conjugate-models

This repository implements Bayesian Conjugate Models in Python.

## 1. Development Environment & Commands

The project uses `uv` for dependency management and command execution.
It is recommended to use `make` commands for common tasks, but underlying `uv` commands are provided for reference.

### Build & Test
- **Run all tests:**
  ```bash
  make test
  # OR
  uv run pytest tests --numprocesses auto
  ```
- **Run a single test:**
  ```bash
  uv run pytest path/to/test_file.py::test_function_name
  ```
- **Run coverage:**
  ```bash
  make cov
  ```
  This runs tests and opens the HTML coverage report.
- **Generate test baselines (visual):**
  ```bash
  make test-generate-baseline
  ```
  Required if visual tests fail due to intentional plotting changes.

### Lint & Format
- **Format & Lint:**
  ```bash
  make format
  ```
  This runs `uv run pre-commit run --all-files`.
- The project uses `ruff` for both linting and formatting via pre-commit hooks.
- **Fix lint errors:**
  ```bash
  ruff check . --fix
  ```

### Documentation
- **Serve docs locally:**
  ```bash
  make html
  ```
  Runs `uv run mkdocs serve`.
- **Edit explorer notebook:**
  ```bash
  make explorer
  ```
  Uses `marimo` to edit the interactive notebook at `docs/explorer.py`.

## 2. Code Style & Conventions

### Formatting & Structure
- **Docstrings:** Use **Google Style** docstrings with **markdown code blocks** for examples.
  ```python
  def function(arg1: int) -> int:
      """Description of function.

      Args:
          arg1: Description of arg1.

      Returns:
          Description of return value.

      Example:
          ```python
          from conjugate.models import some_model
          from conjugate.helpers import some_helper

          data = [1, 2, 3, 4]
          inputs = some_helper(data)
          # inputs = {'x': 10, 'n': 4}
          result = some_model(**inputs, prior=prior)
          ```
      """
  ```
- **CRITICAL Docstring Rule:** Never use doctest format (`>>> `) in docstrings. Always use markdown code blocks with triple backticks and `python` language specification.
  - ❌ **WRONG:** `>>> from conjugate import something`
  - ✅ **CORRECT:**
    ```python
    ```python
    from conjugate import something
    ```
    ```
- **Documentation Rendering:** All docstrings are rendered in MkDocs documentation. Use markdown formatting for proper display:
  - Code examples: Use `\`\`\`python` code blocks
  - Inline code: Use single backticks
  - Parameters: Use standard Google style Args/Returns sections
- **Line Length:** Follow `ruff` defaults (typically 88 chars).
- **Imports:** Sorted by `ruff`. Grouping: Standard lib -> Third party -> Local.
- **File Encoding:** Default to UTF-8.

### Naming Conventions
- **Classes:** `CapWords` (e.g., `Beta`, `Normal`, `Gamma`).
- **Functions/Variables:** `snake_case` (e.g., `plot_pmf`, `daily_rate`, `update_prior`).
- **Constants:** `UPPER_CASE` (if any).
- **Private/Internal:** Prefix with `_` (e.g., `_beta_geometric.py`, `_typing.py`).
- **Mixins:** Suffix with `Mixin` (e.g., `PlotDistMixin`).

### Typing
- The codebase uses type hints extensively (e.g., `from typing import Any, Callable`, `conjugate._typing`).
- Maintain type hints for all new code.
- Use types from `conjugate._typing` where applicable:
  - `NUMERIC`: For array-like or scalar numeric inputs.
  - `Real`, `PositiveReal`: For scalar bounds.
  - `Probability`: For values between 0 and 1.
- While `mypy` is not strictly enforced in the makefile, ensure types are logically consistent.

### Error Handling
- Use standard Python exceptions (e.g., `ValueError`, `TypeError`).
- Use `warnings` for non-critical issues (e.g., parameter deprecation).
- Check `conjugate/models.py` for examples of input validation (often delegating to `scipy.stats`).

### Visual Testing
- This project uses `pytest-mpl` for visual regression testing.
- Tests often look like:
  ```python
  @pytest.mark.mpl_image_compare(filename="test_example.png")
  def test_plot_example():
      fig, ax = plt.subplots()
      # ... plotting code ...
      return fig
  ```
- If you modify plotting logic, you **must** update the baselines using `make test-generate-baseline`.

## 3. Project Structure
- `conjugate/`: Source code.
  - `distributions.py`: Distribution classes (wrapping `scipy.stats` distributions).
  - `models.py`: Conjugate model implementations.
  - `plot.py`: Plotting utilities.
  - `interactive.py`: Interactive elements (likely for documentation).
  - `_typing.py`: Internal type definitions.
- `docs/`: Documentation (Markdown + examples).
  - `examples/`: Usage examples.
- `tests/`: Unit and visual tests.
  - `example-plots/`: Baseline images for `pytest-mpl`.
- `scripts/`: Helper scripts (e.g., `parameter-recovery.py`).

## 4. Development Workflow

### Adding a New Distribution
1.  Define the distribution class in `conjugate/distributions.py` (or a new file if appropriate).
2.  Inherit from appropriate mixins (e.g., `PlotDistMixin`).
3.  Implement methods wrapping `scipy.stats` functionality where possible.
4.  Add typing.
5.  Add unit tests in `tests/test_distributions.py`.

### Adding a New Conjugate Model
1.  Define the model in `conjugate/models.py`.
2.  Implement the `pdf`, `cdf`, `pmf` methods via the `dist` property.
3.  Implement the `posterior` update logic (often an `__add__` or `update` method).
4.  Ensure it works with `conjugate` types.

### Pull Requests
- Ensure `make test` passes locally.
- Ensure `make format` passes (no linting errors).
- Update documentation if public API changes.
- Add examples in `docs/examples/` if a new major feature is added.

## 5. Troubleshooting
- **Visual Tests Failing:**
  - Check if the failure is due to a real regression or a cosmetic change.
  - If cosmetic and desired, run `make test-generate-baseline`.
- **Pre-commit Failures:**
  - Run `make format` to auto-fix most issues.
  - Manually fix remaining issues (often complex logic or variable usage).
- **Environment Issues:**
  - Ensure you are running commands with `uv run` if not using `make`.
  - Check `uv sync` to ensure dependencies are up to date.
