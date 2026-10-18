"""Structured authoritative state for the LangGraph diagnosis loop."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from poise_rca.domain import (
    AnomalyFinding,
    CandidateProposal,
    CaseContext,
    EvidenceRecord,
    Hypothesis,
    InvestigationResult,
    InvestigationTask,
    QueryStatus,
)
from poise_rca.topology import ObservedServiceGraph

RawInvestigationResult = Annotated[InvestigationResult, SkipValidation]


class DiagnosisState(BaseModel):
    """State shared between deterministic graph nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case: CaseContext
    service_inventory: frozenset[str] = frozenset()
    topology: ObservedServiceGraph
    initial_findings: dict[str, AnomalyFinding] = Field(default_factory=dict)
    evidence: dict[str, EvidenceRecord]
    hypotheses: dict[str, Hypothesis]
    tasks: dict[str, InvestigationTask]
    results: tuple[RawInvestigationResult, ...]
    pending_candidate_proposals: tuple[CandidateProposal, ...] = ()
    candidate_rejections: tuple[str, ...] = ()
    invalid_conclusion_task_ids: frozenset[str] = frozenset()
    supervisor_steps: int = Field(default=0, ge=0)
    max_supervisor_steps: int = Field(default=10, gt=0)
    failed_query_signatures: frozenset[str] = frozenset()
    query_statuses: dict[str, QueryStatus] = Field(default_factory=dict)
    unavailable_tools: frozenset[str] = frozenset()
    conflicts: tuple[str, ...]


class StateUpdate(BaseModel):
    """Validated serialized merge output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service_inventory: frozenset[str] = frozenset()
    evidence: dict[str, EvidenceRecord]
    hypotheses: dict[str, Hypothesis]
    tasks: dict[str, InvestigationTask]
    results: tuple[RawInvestigationResult, ...]
    pending_candidate_proposals: tuple[CandidateProposal, ...] = ()
    candidate_rejections: tuple[str, ...] = ()
    invalid_conclusion_task_ids: frozenset[str] = frozenset()
    supervisor_steps: int = Field(default=0, ge=0)
    max_supervisor_steps: int = Field(default=10, gt=0)
    failed_query_signatures: frozenset[str]
    query_statuses: dict[str, QueryStatus]
    conflicts: tuple[str, ...]
