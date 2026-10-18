"""Label-safe incident context exposed to diagnostic agents."""

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator

from poise_rca.domain.enums import Modality

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CaseContext(BaseModel):
    """Only incident inputs that are safe to place in graph state or prompts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: NonEmptyStr
    incident_start: datetime
    incident_end: datetime
    initial_symptom: NonEmptyStr
    entry_service: NonEmptyStr | None = None
    available_modalities: frozenset[Modality] = frozenset()

    @field_validator("incident_start", "incident_end")
    @classmethod
    def timestamps_are_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("incident timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def incident_window_is_ordered(self) -> Self:
        if self.incident_end <= self.incident_start:
            raise ValueError("incident_end must be later than incident_start")
        return self
