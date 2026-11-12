"""Check the Python and package versions used by the project."""
from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def pinned_requirements(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    """Read this project's exact pins, including repository-local -r files."""
    seen = set() if seen is None else seen
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Requirements file is outside the repository: {path}")
    if path in seen:
        raise ValueError(f"Repeated requirements include: {path.relative_to(ROOT)}")
    seen.add(path)
    pins: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            additions = pinned_requirements(path.parent / line[3:].strip(), seen)
        else:
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
            if match is None:
                raise ValueError(f"Expected an exact pin in {path.name}:{number}")
            additions = {match[1]: match[2]}
        for name, version in additions.items():
            if name in pins and pins[name] != version:
                raise ValueError(f"Conflicting pins for {name}")
            pins[name] = version
    return pins


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 13):
        print("Expected Python 3.13.", file=sys.stderr)
        return 1
    try:
        pins = pinned_requirements(ROOT / "requirements-notebook.txt")
    except (OSError, ValueError) as error:
        print(f"Requirements check failed: {error}", file=sys.stderr)
        return 1
    if not pins:
        print("No package pins found.", file=sys.stderr)
        return 1

    failures = []
    for name, expected in sorted(pins.items()):
        try:
            installed = metadata.version(name)
            if installed != expected:
                failures.append(f"{name}: expected {expected}, found {installed}")
                continue
            importlib.import_module({"scikit-learn": "sklearn", "faiss-cpu": "faiss"}.get(name, name))
            print(f"{name}=={installed}: import OK")
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print("Environment checks passed; no model or service tests were run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
