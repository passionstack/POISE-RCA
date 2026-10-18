"""Static anomaly-analysis contracts used to seed diagnosis."""

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints, field_validator

from poise_rca.domain.enums import Modality, QueryStatus

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_uuid_string(value: str) -> str:
    UUID(value)
    return value


class AnomalyFinding(BaseModel):
    """One ranked, provenance-preserving anomaly produced during initialization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str
    service: NonEmptyStr
    modality: Modality
    signal: NonEmptyStr
    observation: NonEmptyStr
    source_reference: NonEmptyStr
    analyzer: NonEmptyStr
    score: float | None = None
    rank: PositiveInt | None = None

    _finding_id_is_uuid = field_validator("finding_id")(_validate_uuid_string)


class StaticAnalysisResult(BaseModel):
    """Findings plus explicit per-modality analysis outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: tuple[AnomalyFinding, ...] = ()
    statuses: dict[Modality, QueryStatus]
    errors: dict[Modality, NonEmptyStr] = Field(default_factory=dict)
    unmapped_indicators: tuple[NonEmptyStr, ...] = ()
