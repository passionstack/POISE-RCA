"""Dataset-side models separated from label-safe diagnostic state."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from poise_rca.domain import CaseContext, Modality

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SafeCaseId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^re2-ob-[0-9a-f]{16}$",
    ),
]


class CaseRecord(BaseModel):
    """Offline discovery record; labels must never be passed to an agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    safe_case_id: SafeCaseId
    source_locator: Path = Field(repr=False)
    dataset_version: NonEmptyStr
    root_cause_service: NonEmptyStr
    fault_type: NonEmptyStr
    instance: int = Field(ge=1)
    checksums: dict[str, str]


@dataclass(frozen=True, slots=True)
class TelemetryBundle:
    """Loaded telemetry without labels or a source path."""

    safe_case_id: str
    incident_start: datetime
    incident_end: datetime
    initial_symptom: str
    entry_service: str | None
    metrics: pd.DataFrame | None
    logs: pd.DataFrame | None
    traces: pd.DataFrame | None

    @property
    def available_modalities(self) -> frozenset[Modality]:
        modalities: set[Modality] = set()
        if self.metrics is not None:
            modalities.add(Modality.METRICS)
        if self.logs is not None:
            modalities.add(Modality.LOGS)
        if self.traces is not None:
            modalities.add(Modality.TRACES)
        return frozenset(modalities)

    def agent_view(self) -> CaseContext:
        """Return the only case representation allowed in graph state and prompts."""

        return CaseContext(
            case_id=self.safe_case_id,
            incident_start=self.incident_start,
            incident_end=self.incident_end,
            initial_symptom=self.initial_symptom,
            entry_service=self.entry_service,
            available_modalities=self.available_modalities,
        )
