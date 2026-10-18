"""Semantic diagnosis decisions and the supervisor-loop safety stop."""

from pydantic import BaseModel, ConfigDict

from poise_rca.domain import (
    CausalRole,
    DiagnosisStatus,
    HypothesisStatus,
    InvestigationResult,
    QueryStatus,
    TaskStatus,
)
from poise_rca.graph.state import DiagnosisState


class DiagnosisDecision(BaseModel):
    """Auditable stop decision made without budgets or invented evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: DiagnosisStatus
    stop_reason: str
    root_cause_hypothesis_id: str | None = None
    remaining_hypothesis_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    next_human_action: str | None = None


def _missing_evidence(state: DiagnosisState) -> tuple[str, ...]:
    return tuple(
        sorted(
            {gap for hypothesis in state.hypotheses.values() for gap in hypothesis.missing_evidence}
        )
    )


def _valid_results(state: DiagnosisState) -> tuple[InvestigationResult, ...]:
    return tuple(
        result
        for result in state.results
        if result.task_id not in state.invalid_conclusion_task_ids and result.conclusion is not None
    )


def _unresolved_questions(state: DiagnosisState) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            question for result in _valid_results(state) for question in result.unresolved_questions
        )
    )


def _limitations(state: DiagnosisState) -> tuple[str, ...]:
    limiting_statuses = {
        QueryStatus.SUCCESS_EMPTY,
        QueryStatus.PARTIAL,
        QueryStatus.DATA_UNAVAILABLE,
        QueryStatus.TOOL_ERROR,
        QueryStatus.TIMEOUT,
    }
    return tuple(
        sorted(
            f"query {signature}: {status.value}"
            for signature, status in state.query_statuses.items()
            if status in limiting_statuses
        )
    )


def _leader_is_grounded(state: DiagnosisState, hypothesis_id: str) -> bool:
    hypothesis = state.hypotheses[hypothesis_id]
    return any(
        (record := state.evidence.get(identifier)) is not None
        and record.query_status is QueryStatus.SUCCESS_WITH_DATA
        and record.attributes.get("causal_support") is True
        and record.attributes.get("comparative") is True
        for identifier in hypothesis.supporting_evidence_ids
    )


def evaluate_stop(state: DiagnosisState) -> DiagnosisDecision:
    """Classify current evidence semantically; only supervisor steps bound replanning."""

    concrete = tuple(
        hypothesis
        for hypothesis in state.hypotheses.values()
        if hypothesis.root_cause_service != "OTHER/UNKNOWN"
    )
    plausible = tuple(
        sorted(
            hypothesis.hypothesis_id
            for hypothesis in concrete
            if hypothesis.status is not HypothesisStatus.REFUTED
        )
    )
    missing = _missing_evidence(state)
    limitations = _limitations(state)
    unresolved = _unresolved_questions(state)
    open_tasks = any(
        task.status in {TaskStatus.PENDING, TaskStatus.RUNNING} for task in state.tasks.values()
    )
    leaders = [
        hypothesis for hypothesis in concrete if hypothesis.status is HypothesisStatus.LEADING
    ]
    alternatives = [
        hypothesis
        for hypothesis in concrete
        if hypothesis.status in {HypothesisStatus.ACTIVE, HypothesisStatus.DEFERRED}
    ]

    if (
        len(leaders) == 1
        and not alternatives
        and not open_tasks
        and not state.pending_candidate_proposals
        and _leader_is_grounded(state, leaders[0].hypothesis_id)
    ):
        return DiagnosisDecision(
            status=DiagnosisStatus.CONFIRMED,
            stop_reason="GROUNDED_LEADER_WITH_REFUTED_ALTERNATIVES",
            root_cause_hypothesis_id=leaders[0].hypothesis_id,
            limitations=limitations,
            unresolved_questions=unresolved,
        )

    if state.supervisor_steps >= state.max_supervisor_steps:
        return DiagnosisDecision(
            status=DiagnosisStatus.INDETERMINATE,
            stop_reason="MAX_SUPERVISOR_STEPS",
            remaining_hypothesis_ids=plausible,
            missing_evidence=missing,
            limitations=limitations,
            unresolved_questions=unresolved,
            next_human_action="Review unresolved candidates and collect discriminating telemetry.",
        )

    failure_statuses = {
        QueryStatus.DATA_UNAVAILABLE,
        QueryStatus.PARTIAL,
        QueryStatus.TOOL_ERROR,
        QueryStatus.TIMEOUT,
    }
    if not _valid_results(state) and any(
        status in failure_statuses for status in state.query_statuses.values()
    ):
        return DiagnosisDecision(
            status=DiagnosisStatus.OBSERVATION_FAILURE,
            stop_reason="OBSERVATION_FAILURE_BEFORE_COMPARISON",
            remaining_hypothesis_ids=plausible,
            missing_evidence=missing,
            limitations=limitations,
            unresolved_questions=unresolved,
            next_human_action="Restore the failed or unavailable telemetry source.",
        )

    narrowed = bool(leaders) or any(
        hypothesis.status is HypothesisStatus.REFUTED
        or hypothesis.causal_role is CausalRole.PROPAGATED_SYMPTOM
        for hypothesis in concrete
    )
    if narrowed:
        return DiagnosisDecision(
            status=DiagnosisStatus.NARROWED,
            stop_reason="CANDIDATE_SET_NARROWED_WITH_REMAINING_UNCERTAINTY",
            remaining_hypothesis_ids=plausible,
            missing_evidence=missing,
            limitations=limitations,
            unresolved_questions=unresolved,
            next_human_action="Resolve the remaining discriminating observations.",
        )
    return DiagnosisDecision(
        status=DiagnosisStatus.INDETERMINATE,
        stop_reason="PLAUSIBLE_CANDIDATES_NOT_DISTINGUISHED",
        remaining_hypothesis_ids=plausible,
        missing_evidence=missing,
        limitations=limitations,
        unresolved_questions=unresolved,
        next_human_action="Collect an observation that separates the remaining candidates.",
    )
