#!/usr/bin/env python3
"""Check that markdown files in docs/ have required frontmatter.

This script validates that all markdown files in the docs/ directory
(except docs/index.md) have YAML frontmatter containing 'comments: true'.

This is used as a pre-commit hook to ensure documentation pages have
comments enabled for user engagement.

Usage:
    python scripts/check_frontmatter.py
"""

from pathlib import Path
import re
import sys

# Files to exclude from validation
EXCLUDED_FILES = {
    Path("docs/index.md"),
}

# Pattern to match YAML frontmatter at the start of a file
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)

# Pattern to match 'comments: true' (case-insensitive for the boolean)
COMMENTS_PATTERN = re.compile(r"^comments:\s*(true|True)$", re.MULTILINE)


def check_frontmatter(file_path: Path) -> str | None:
    """Check if a file has valid frontmatter with comments: true.

    Args:
        file_path: Path to the markdown file to check.

    Returns:
        Error message if validation fails, None if validation passes.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Could not read file: {e}"

    # Check for frontmatter
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return "Missing YAML frontmatter (file should start with '---')"

    frontmatter = match.group(1)

    # Check for comments: true
    if not COMMENTS_PATTERN.search(frontmatter):
        return "Frontmatter missing 'comments: true'"

    return None


def main() -> int:
    """Run frontmatter validation on all docs markdown files.

    Returns:
        Exit code: 0 if all files pass, 1 if any fail.
    """
    docs_dir = Path("docs")

    if not docs_dir.exists():
        print("Error: docs/ directory not found", file=sys.stderr)
        return 1

    # Find all markdown files in docs/
    md_files = sorted(docs_dir.rglob("*.md"))

    # Filter out excluded files
    files_to_check = [f for f in md_files if f.relative_to(".") not in EXCLUDED_FILES]

    errors: list[tuple[Path, str]] = []

    for file_path in files_to_check:
        error = check_frontmatter(file_path)
        if error:
            errors.append((file_path, error))

    if errors:
        print("Frontmatter validation failed:", file=sys.stderr)
        print(file=sys.stderr)
        for file_path, error in errors:
            print(f"  {file_path}: {error}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "All markdown files in docs/ (except index.md) must have "
            "frontmatter with 'comments: true'.",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print("Example frontmatter:", file=sys.stderr)
        print("  ---", file=sys.stderr)
        print("  comments: true", file=sys.stderr)
        print("  ---", file=sys.stderr)
        return 1

    print(f"Frontmatter validation passed: {len(files_to_check)} files checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
