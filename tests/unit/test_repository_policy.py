from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_artifacts_are_gitignored() -> None:
    patterns = {
        line.strip()
        for line in (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    required_patterns = {
        ".env",
        "data/raw/",
        "data/processed/",
        "results/",
        "logs/",
        "checkpoints/",
        ".venv/",
        "*.bin",
        "*.safetensors",
        "*.sqlite",
        "*.sqlite3",
        ".langfuse/",
        "local-notes/",
        "!data/manifests/*.json",
    }

    assert required_patterns <= patterns
