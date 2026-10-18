"""Diagnostic hypothesis contracts and referential integrity checks."""

from collections.abc import Mapping
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator

from poise_rca.domain.enums import CausalRole, HypothesisStatus
from poise_rca.domain.evidence import EvidenceRecord

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_uuid_string(value: str) -> str:
    UUID(value)
    return value


def _validate_uuid_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    for value in values:
        _validate_uuid_string(value)
    return values


class Hypothesis(BaseModel):
    """A retained, evidence-addressable candidate causal explanation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis_id: str
    root_cause_service: NonEmptyStr
    fault_type: NonEmptyStr
    causal_claim: NonEmptyStr
    causal_role: CausalRole = CausalRole.UNKNOWN
    origin_evidence_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    expected_observations: tuple[NonEmptyStr, ...] = ()
    missing_evidence: tuple[NonEmptyStr, ...] = ()
    competing_hypothesis_ids: tuple[str, ...] = ()
    status: HypothesisStatus = HypothesisStatus.ACTIVE

    _hypothesis_id_is_uuid = field_validator("hypothesis_id")(_validate_uuid_string)
    _evidence_ids_are_uuids = field_validator(
        "origin_evidence_ids",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "competing_hypothesis_ids",
    )(_validate_uuid_tuple)

    @model_validator(mode="after")
    def has_testable_or_missing_observation(self) -> Self:
        resolved_evidence = self.supporting_evidence_ids or self.contradicting_evidence_ids
        if not self.expected_observations and not self.missing_evidence and not resolved_evidence:
            raise ValueError("a hypothesis requires an expected observation or missing evidence")
        if self.status is HypothesisStatus.REFUTED and not self.contradicting_evidence_ids:
            raise ValueError("a refuted hypothesis must retain contradicting evidence")
        return self


def validate_evidence_links(
    hypotheses: Mapping[str, Hypothesis],
    evidence: Mapping[str, EvidenceRecord],
) -> None:
    """Reject dangling evidence and competitor references in hypothesis state."""

    evidence_ids = set(evidence)
    hypothesis_ids = set(hypotheses)
    for key, hypothesis in hypotheses.items():
        if key != hypothesis.hypothesis_id:
            raise ValueError(f"hypothesis key {key!r} does not match its ID")
        linked_evidence = (
            set(hypothesis.origin_evidence_ids)
            | set(hypothesis.supporting_evidence_ids)
            | set(hypothesis.contradicting_evidence_ids)
        )
        missing_evidence_ids = linked_evidence - evidence_ids
        if missing_evidence_ids:
            raise ValueError(
                f"hypothesis {key!r} references missing evidence: {sorted(missing_evidence_ids)}"
            )
        missing_competitors = set(hypothesis.competing_hypothesis_ids) - hypothesis_ids
        if missing_competitors:
            raise ValueError(
                f"hypothesis {key!r} references missing competitors: {sorted(missing_competitors)}"
            )
