"""Immutable evidence-ledger contracts and provenance validation."""

from datetime import datetime
from typing import Annotated, Any, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from poise_rca.domain.enums import EvidencePolarity, Modality, QueryStatus

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_uuid_string(value: str) -> str:
    UUID(value)
    return value


class EvidenceRecord(BaseModel):
    """One append-only normalized observation with its local-data provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    modality: Modality
    service: NonEmptyStr
    observation: NonEmptyStr
    query_id: str
    query_status: QueryStatus
    polarity: EvidencePolarity = EvidencePolarity.NEUTRAL
    source_reference: NonEmptyStr
    parent_evidence_ids: tuple[str, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    _evidence_id_is_uuid = field_validator("evidence_id")(_validate_uuid_string)
    _query_id_is_uuid = field_validator("query_id")(_validate_uuid_string)

    @field_validator("parent_evidence_ids")
    @classmethod
    def validate_parent_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _validate_uuid_string(value)
        return values

    @field_validator("created_at")
    @classmethod
    def created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def valid_polarity_for_query_outcome(self) -> Self:
        if (
            self.polarity in {EvidencePolarity.SUPPORT, EvidencePolarity.CONTRADICT}
            and self.query_status is not QueryStatus.SUCCESS_WITH_DATA
        ):
            raise ValueError("supporting or contradicting evidence requires SUCCESS_WITH_DATA")
        return self
