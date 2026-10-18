"""Structured supervisor planning with deterministic safety validation."""

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from typing import Annotated, Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from poise_rca.agents.hypothesis_generator import EvidencePromptSummary
from poise_rca.domain import (
    CandidateProposal,
    CausalRole,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    InvestigationResult,
    InvestigationTask,
    Modality,
    TaskKind,
)
from poise_rca.topology import ObservedServiceGraph

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SupervisorHypothesisSummary(BaseModel):
    """Label-safe hypothesis projection supplied to the supervisor model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis_id: str
    root_cause_service: str
    fault_type: str
    causal_claim: str
    causal_role: CausalRole
    status: HypothesisStatus
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    expected_observations: tuple[str, ...]
    missing_evidence: tuple[str, ...]


class CompletedInvestigationSummary(BaseModel):
    """Global-result projection without a worker's private conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    conclusion: str | None
    cited_evidence_ids: tuple[str, ...]
    unresolved_questions: tuple[str, ...]


class SupervisorPromptInput(BaseModel):
    """Safe, structured global state visible during one supervisor step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypotheses: tuple[SupervisorHypothesisSummary, ...]
    evidence: tuple[EvidencePromptSummary, ...]
    completed_investigations: tuple[CompletedInvestigationSummary, ...] = ()
    completed_task_signatures: tuple[str, ...] = ()
    pending_proposals: tuple[CandidateProposal, ...] = ()
    topology: ObservedServiceGraph
    available_modalities: frozenset[Modality]


class InvestigationTaskDraft(BaseModel):
    """A model-proposed task that has not yet passed deterministic validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_kind: TaskKind
    target_hypothesis_ids: tuple[str, ...]
    question: NonEmptyStr
    shared_evidence_ids: tuple[str, ...] = ()
    discriminating_observations: tuple[NonEmptyStr, ...]
    available_modalities: tuple[Modality, ...]


class HypothesisTransitionRecommendation(BaseModel):
    """Evidence-citing status advice; reducers remain responsible for mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis_id: str
    target_status: HypothesisStatus
    rationale: NonEmptyStr
    evidence_ids: tuple[str, ...]

    @field_validator("evidence_ids")
    @classmethod
    def cites_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("a transition recommendation must cite evidence")
        return values


class SupervisorPlan(BaseModel):
    """Structured task and lifecycle recommendations authored by the supervisor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: tuple[InvestigationTaskDraft, ...] = ()
    transition_recommendations: tuple[HypothesisTransitionRecommendation, ...] = ()
    no_meaningful_competition: bool = False
    rationale: str | None = None


class SupervisorModel(Protocol):
    """Minimal structured model boundary for global planning."""

    def invoke(self, input: SupervisorPromptInput) -> object:
        """Return an object parseable as a supervisor plan."""


def _normalize_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def task_signature(task: InvestigationTaskDraft | InvestigationTask) -> str:
    """Return the canonical identity of an investigation independent of display order."""

    parts = (
        task.task_kind.value,
        ",".join(sorted(task.target_hypothesis_ids)),
        _normalize_question(task.question),
        ",".join(sorted(task.shared_evidence_ids)),
    )
    return "|".join(parts)


def _task_id(draft: InvestigationTaskDraft) -> str:
    signature = task_signature(draft)
    return str(uuid5(NAMESPACE_URL, f"poise-rca:investigation:{signature}"))


def _valid_targets(
    draft: InvestigationTaskDraft,
    hypotheses: Mapping[str, Hypothesis],
) -> bool:
    targets = draft.target_hypothesis_ids
    expected_count = 2 if draft.task_kind is TaskKind.COMPARISON else 1
    if len(targets) != expected_count or len(set(targets)) != expected_count:
        return False
    selected = [hypotheses.get(identifier) for identifier in targets]
    if any(item is None or item.status is HypothesisStatus.REFUTED for item in selected):
        return False
    includes_other_unknown = draft.task_kind is TaskKind.COMPARISON and any(
        item is not None and item.root_cause_service.casefold() == "other/unknown"
        for item in selected
    )
    return not includes_other_unknown


def _compares_compatible_layers(
    draft: InvestigationTaskDraft,
    hypotheses: Mapping[str, Hypothesis],
) -> bool:
    if draft.task_kind is not TaskKind.COMPARISON:
        return False
    roles = {hypotheses[identifier].causal_role for identifier in draft.target_hypothesis_ids}
    roles.discard(CausalRole.UNKNOWN)
    return len(roles) > 1


def validate_supervisor_plan(
    plan: SupervisorPlan,
    hypotheses: Mapping[str, Hypothesis],
    evidence: Mapping[str, EvidenceRecord],
    *,
    completed_task_signatures: Collection[str] = (),
    available_modalities: Collection[Modality],
    incident_start: datetime,
    incident_end: datetime,
) -> tuple[InvestigationTask, ...]:
    """Commit only well-shaped, grounded, novel tasks from a model-authored plan."""

    accepted: list[InvestigationTask] = []
    seen = set(completed_task_signatures)
    evidence_ids = set(evidence)
    modality_set = set(available_modalities)
    for draft in plan.tasks:
        signature = task_signature(draft)
        if signature in seen:
            continue
        if not _valid_targets(draft, hypotheses):
            continue
        if not set(draft.shared_evidence_ids) <= evidence_ids:
            continue
        if not draft.discriminating_observations:
            continue
        if not draft.available_modalities or not set(draft.available_modalities) <= modality_set:
            continue
        if _compares_compatible_layers(draft, hypotheses) and (
            "initiating cause" not in _normalize_question(draft.question)
        ):
            continue
        accepted.append(
            InvestigationTask(
                task_id=_task_id(draft),
                task_kind=draft.task_kind,
                target_hypothesis_ids=draft.target_hypothesis_ids,
                question=draft.question,
                shared_evidence_ids=draft.shared_evidence_ids,
                discriminating_observations=draft.discriminating_observations,
                available_modalities=draft.available_modalities,
                start_time=incident_start,
                end_time=incident_end,
            )
        )
        seen.add(signature)
    return tuple(accepted)


_LEGAL_TRANSITIONS: dict[HypothesisStatus, frozenset[HypothesisStatus]] = {
    HypothesisStatus.ACTIVE: frozenset(
        {HypothesisStatus.DEFERRED, HypothesisStatus.REFUTED, HypothesisStatus.LEADING}
    ),
    HypothesisStatus.DEFERRED: frozenset({HypothesisStatus.ACTIVE, HypothesisStatus.REFUTED}),
    HypothesisStatus.REFUTED: frozenset({HypothesisStatus.ACTIVE}),
    HypothesisStatus.LEADING: frozenset(
        {HypothesisStatus.ACTIVE, HypothesisStatus.DEFERRED, HypothesisStatus.REFUTED}
    ),
}


def validate_transition_recommendations(
    recommendations: Sequence[HypothesisTransitionRecommendation],
    hypotheses: Mapping[str, Hypothesis],
    evidence: Mapping[str, EvidenceRecord],
) -> tuple[HypothesisTransitionRecommendation, ...]:
    """Filter ungrounded or illegal advice without changing hypothesis state."""

    accepted: list[HypothesisTransitionRecommendation] = []
    evidence_ids = set(evidence)
    for recommendation in recommendations:
        hypothesis = hypotheses.get(recommendation.hypothesis_id)
        if hypothesis is None:
            continue
        if not set(recommendation.evidence_ids) <= evidence_ids:
            continue
        if recommendation.target_status not in _LEGAL_TRANSITIONS[hypothesis.status]:
            continue
        accepted.append(recommendation)
    return tuple(accepted)


def plan_investigations(
    hypotheses: Mapping[str, Hypothesis],
    evidence: Mapping[str, EvidenceRecord],
    completed_investigations: Sequence[InvestigationResult],
    completed_task_signatures: Collection[str],
    pending_proposals: Sequence[CandidateProposal],
    topology: ObservedServiceGraph,
    available_modalities: Collection[Modality],
    model: SupervisorModel,
) -> SupervisorPlan:
    """Ask the supervisor for a structured plan using only safe global projections."""

    prompt_input = SupervisorPromptInput(
        hypotheses=tuple(
            _hypothesis_summary(item)
            for item in sorted(hypotheses.values(), key=lambda value: value.hypothesis_id)
        ),
        evidence=tuple(
            _evidence_summary(item)
            for item in sorted(evidence.values(), key=lambda value: value.evidence_id)
        ),
        completed_investigations=tuple(
            CompletedInvestigationSummary(
                task_id=result.task_id,
                conclusion=result.conclusion,
                cited_evidence_ids=result.cited_evidence_ids,
                unresolved_questions=result.unresolved_questions,
            )
            for result in completed_investigations
        ),
        completed_task_signatures=tuple(sorted(completed_task_signatures)),
        pending_proposals=tuple(pending_proposals),
        topology=topology,
        available_modalities=frozenset(available_modalities),
    )
    return SupervisorPlan.model_validate(model.invoke(prompt_input))


def _hypothesis_summary(hypothesis: Hypothesis) -> SupervisorHypothesisSummary:
    return SupervisorHypothesisSummary(
        hypothesis_id=hypothesis.hypothesis_id,
        root_cause_service=hypothesis.root_cause_service,
        fault_type=hypothesis.fault_type,
        causal_claim=hypothesis.causal_claim,
        causal_role=hypothesis.causal_role,
        status=hypothesis.status,
        supporting_evidence_ids=hypothesis.supporting_evidence_ids,
        contradicting_evidence_ids=hypothesis.contradicting_evidence_ids,
        expected_observations=hypothesis.expected_observations,
        missing_evidence=hypothesis.missing_evidence,
    )


def _evidence_summary(record: EvidenceRecord) -> EvidencePromptSummary:
    return EvidencePromptSummary(
        evidence_id=record.evidence_id,
        modality=record.modality.value,
        service=record.service,
        observation=record.observation,
        query_status=record.query_status.value,
    )
