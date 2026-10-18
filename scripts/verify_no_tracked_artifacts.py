"""Fail when Git tracks runtime, dataset, credential, or model artifacts."""

import subprocess
import sys
from pathlib import PurePosixPath

_BLOCKED_ROOTS = {
    ".langfuse",
    "checkpoints",
    "langfuse-data",
    "logs",
    "models",
    "results",
}
_BLOCKED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".db",
    ".log",
    ".onnx",
    ".parquet",
    ".pem",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
}
_SYNTHETIC_FIXTURE_ROOT = PurePosixPath("tests/fixtures/synthetic_case")


def _allowed(path: PurePosixPath) -> bool:
    if path == PurePosixPath(".env.example"):
        return True
    if path.is_relative_to(_SYNTHETIC_FIXTURE_ROOT):
        return True
    return path.parent == PurePosixPath("data/manifests") and path.suffix.casefold() == ".json"


def _violation(path: PurePosixPath) -> str | None:
    if _allowed(path):
        return None
    parts = {part.casefold() for part in path.parts}
    if parts & _BLOCKED_ROOTS:
        return "runtime/model artifact directory"
    if path.parts and path.parts[0].casefold() == "data":
        return "dataset or derived data"
    if path.name == ".env" or path.name.startswith(".env."):
        return "environment secret file"
    if path.suffix.casefold() in _BLOCKED_SUFFIXES:
        return f"blocked artifact extension {path.suffix}"
    if path.suffix.casefold() == ".csv":
        return "unapproved tabular dataset"
    if path.name.casefold() in {"credentials.json", "secrets.yaml", "secrets.yml"}:
        return "credential file"
    return None


def main() -> int:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    tracked = [PurePosixPath(raw.decode("utf-8")) for raw in completed.stdout.split(b"\0") if raw]
    violations = [(path, reason) for path in tracked if (reason := _violation(path)) is not None]
    if violations:
        print("Tracked artifact policy violations:", file=sys.stderr)
        for path, reason in violations:
            print(f"- {path}: {reason}", file=sys.stderr)
        return 1
    print(f"Checked {len(tracked)} tracked paths; no forbidden artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
