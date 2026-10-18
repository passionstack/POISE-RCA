"""Serializable sampled-dataset manifests with an explicit agent-safe view."""

import re
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

SafeCaseId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^re2-ob-[0-9a-f]{16}$",
    ),
]
_CHECKSUM_FILENAMES = frozenset(
    {
        "metrics.parquet",
        "metrics.json",
        "metrics.csv",
        "logs.parquet",
        "logs.csv",
        "traces.parquet",
        "traces.csv",
        "inject_time.txt",
    }
)


def _validate_checksums(value: dict[str, str]) -> dict[str, str]:
    if any(name not in _CHECKSUM_FILENAMES for name in value):
        raise ValueError("checksums contain an unsupported filename")
    if any(re.fullmatch(r"[0-9a-f]{64}", checksum) is None for checksum in value.values()):
        raise ValueError("checksums must be lowercase SHA-256 hex digests")
    return value


class ManifestCase(BaseModel):
    """Offline case metadata used for reproducibility and later evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    safe_case_id: SafeCaseId
    source_locator: str
    root_cause_service: str
    fault_type: str
    checksums: dict[str, str]

    _checksums_are_safe = field_validator("checksums")(_validate_checksums)


class AgentSafeManifestCase(BaseModel):
    """Manifest entry with private location and labels removed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    safe_case_id: SafeCaseId
    checksums: dict[str, str]

    _checksums_are_safe = field_validator("checksums")(_validate_checksums)


class AgentSafeManifest(BaseModel):
    """Manifest projection safe for orchestration but not used for diagnosis content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str
    seed: int
    cases: tuple[AgentSafeManifestCase, ...]


class DatasetManifest(BaseModel):
    """Deterministic offline selection record; never pass this object to an agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str
    seed: int
    constraints: dict[str, str | int | bool | tuple[str, ...] | None]
    cases: tuple[ManifestCase, ...]

    def agent_safe_view(self) -> AgentSafeManifest:
        return AgentSafeManifest(
            dataset_version=self.dataset_version,
            seed=self.seed,
            cases=tuple(
                AgentSafeManifestCase(
                    safe_case_id=case.safe_case_id,
                    checksums=case.checksums,
                )
                for case in self.cases
            ),
        )


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Write a stable JSON representation of a sampled dataset manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
