"""Supervisor tasks and worker reports."""

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

from poise_rca.domain.enums import Modality, QueryStatus, TaskKind, TaskStatus
from poise_rca.domain.evidence import EvidenceRecord

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_uuid_string(value: str) -> str:
    UUID(value)
    return value


def _validate_uuid_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    for value in values:
        _validate_uuid_string(value)
    return values


def _validate_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class InvestigationTask(BaseModel):
    """One comparative or challenge question assigned to an isolated worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_kind: TaskKind = TaskKind.CHALLENGE
    target_hypothesis_ids: tuple[str, ...]
    question: NonEmptyStr
    shared_evidence_ids: tuple[str, ...] = ()
    discriminating_observations: tuple[NonEmptyStr, ...] = ()
    available_modalities: tuple[Modality, ...] = ()
    modality: Modality | None = None
    service: NonEmptyStr | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    tool_name: NonEmptyStr | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_outcomes: tuple[NonEmptyStr, ...] = ()
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING

    _task_id_is_uuid = field_validator("task_id")(_validate_uuid_string)
    _referenced_ids_are_uuids = field_validator("target_hypothesis_ids", "shared_evidence_ids")(
        _validate_uuid_tuple
    )

    @field_validator("target_hypothesis_ids")
    @classmethod
    def has_target_hypothesis(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("an investigation task requires a target hypothesis")
        return values

    @field_validator("start_time", "end_time")
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        return _validate_aware(value) if value is not None else None

    @model_validator(mode="after")
    def task_shape_is_valid(self) -> Self:
        target_count = len(self.target_hypothesis_ids)
        if self.task_kind is TaskKind.COMPARISON and (
            target_count != 2 or len(set(self.target_hypothesis_ids)) != 2
        ):
            raise ValueError("a comparison requires two distinct hypotheses")
        if self.task_kind is TaskKind.CHALLENGE and target_count != 1:
            raise ValueError("a challenge requires exactly one hypothesis")
        if (self.start_time is None) is not (self.end_time is None):
            raise ValueError("start_time and end_time must be provided together")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError("end_time must be later than start_time")
        return self


class CandidateProposal(BaseModel):
    """Evidence-grounded new hypothesis suggested by a worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: NonEmptyStr
    fault_type: NonEmptyStr
    causal_claim: NonEmptyStr
    evidence_ids: tuple[str, ...]
    suggested_investigation: NonEmptyStr

    _evidence_ids_are_uuids = field_validator("evidence_ids")(_validate_uuid_tuple)

    @field_validator("evidence_ids")
    @classmethod
    def has_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("a candidate proposal requires evidence")
        return values


class InvestigationResult(BaseModel):
    """Evidence-grounded worker conclusion consumed by the serialized merge node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    evidence: tuple[EvidenceRecord, ...] = ()
    conclusion: NonEmptyStr | None = None
    cited_evidence_ids: tuple[str, ...] = ()
    candidate_proposals: tuple[CandidateProposal, ...] = ()
    unresolved_questions: tuple[NonEmptyStr, ...] = ()
    query_signatures: tuple[NonEmptyStr, ...] = ()
    query_statuses: dict[NonEmptyStr, QueryStatus] = Field(default_factory=dict)
    query_id: str | None = None
    status: QueryStatus | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    _task_id_is_uuid = field_validator("task_id")(_validate_uuid_string)
    _cited_ids_are_uuids = field_validator("cited_evidence_ids")(_validate_uuid_tuple)

    @field_validator("query_id")
    @classmethod
    def query_id_is_uuid(cls, value: str | None) -> str | None:
        return _validate_uuid_string(value) if value is not None else None

    @field_validator("started_at", "finished_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        return _validate_aware(value) if value is not None else None

    @model_validator(mode="after")
    def result_is_consistent(self) -> Self:
        if (self.started_at is None) is not (self.finished_at is None):
            raise ValueError("started_at and finished_at must be provided together")
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at cannot precede started_at")
        evidence_ids = {record.evidence_id for record in self.evidence}
        missing_citations = set(self.cited_evidence_ids) - evidence_ids
        if missing_citations:
            raise ValueError(f"conclusion cites missing evidence: {sorted(missing_citations)}")
        return self
