from datetime import UTC, datetime, timedelta
from uuid import uuid4

from poise_rca.domain import (
    CandidateProposal,
    CaseContext,
    CausalRole,
    DiagnosisStatus,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    TaskKind,
)
from poise_rca.graph.state import DiagnosisState
from poise_rca.graph.stopping import evaluate_stop
from poise_rca.topology import build_observed_service_graph


def _evidence(
    *,
    service: str = "checkout",
    support: bool = True,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.TRACES,
        service=service,
        observation=f"{service} anomaly precedes downstream latency.",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.SUPPORT if support else EvidencePolarity.NEUTRAL,
        source_reference=f"trace:{service}",
        attributes={"causal_support": support, "comparative": support},
        created_at=datetime.now(UTC),
    )


def _hypothesis(
    service: str,
    status: HypothesisStatus,
    evidence_id: str,
    *,
    role: CausalRole = CausalRole.UNKNOWN,
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service=service,
        fault_type="delay",
        causal_claim=f"{service} delay initiated the incident.",
        causal_role=role,
        supporting_evidence_ids=(evidence_id,) if status is HypothesisStatus.LEADING else (),
        contradicting_evidence_ids=(evidence_id,) if status is HypothesisStatus.REFUTED else (),
        expected_observations=(f"{service} anomaly occurs first",),
        status=status,
    )


def _state(
    hypotheses: tuple[Hypothesis, ...],
    evidence: tuple[EvidenceRecord, ...],
    *,
    results: tuple[InvestigationResult, ...] = (),
    tasks: tuple[InvestigationTask, ...] = (),
    proposals: tuple[CandidateProposal, ...] = (),
    query_statuses: dict[str, QueryStatus] | None = None,
    supervisor_steps: int = 1,
    max_supervisor_steps: int = 10,
) -> DiagnosisState:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return DiagnosisState(
        case=CaseContext(
            case_id="re2-ob-0000000000000001",
            incident_start=start,
            incident_end=start + timedelta(minutes=5),
            initial_symptom="Elevated latency.",
            available_modalities=frozenset({Modality.TRACES}),
        ),
        service_inventory=frozenset(item.root_cause_service for item in hypotheses),
        topology=build_observed_service_graph(evidence),
        evidence={item.evidence_id: item for item in evidence},
        hypotheses={item.hypothesis_id: item for item in hypotheses},
        tasks={item.task_id: item for item in tasks},
        results=results,
        pending_candidate_proposals=proposals,
        query_statuses=query_statuses or {},
        conflicts=(),
        supervisor_steps=supervisor_steps,
        max_supervisor_steps=max_supervisor_steps,
    )


def test_one_grounded_leader_with_only_refuted_alternatives_confirms() -> None:
    evidence = _evidence()
    leader = _hypothesis("checkout", HypothesisStatus.LEADING, evidence.evidence_id)
    alternative = _hypothesis("payment", HypothesisStatus.REFUTED, evidence.evidence_id)

    decision = evaluate_stop(_state((leader, alternative), (evidence,)))

    assert decision.status is DiagnosisStatus.CONFIRMED
    assert decision.root_cause_hypothesis_id == leader.hypothesis_id


def test_deferred_plausible_alternative_prevents_strict_confirmation() -> None:
    evidence = _evidence()
    leader = _hypothesis("checkout", HypothesisStatus.LEADING, evidence.evidence_id)
    deferred = _hypothesis("payment", HypothesisStatus.DEFERRED, evidence.evidence_id)

    decision = evaluate_stop(_state((leader, deferred), (evidence,)))

    assert decision.status is DiagnosisStatus.NARROWED
    assert decision.root_cause_hypothesis_id is None
    assert deferred.hypothesis_id in decision.remaining_hypothesis_ids


def test_multiple_active_candidates_are_indeterminate_or_narrowed_by_causal_role() -> None:
    evidence = _evidence(support=False)
    first = _hypothesis("checkout", HypothesisStatus.ACTIVE, evidence.evidence_id)
    second = _hypothesis("payment", HypothesisStatus.ACTIVE, evidence.evidence_id)
    propagated = _hypothesis(
        "frontend",
        HypothesisStatus.ACTIVE,
        evidence.evidence_id,
        role=CausalRole.PROPAGATED_SYMPTOM,
    )

    indeterminate = evaluate_stop(_state((first, second), (evidence,)))
    narrowed = evaluate_stop(_state((first, second, propagated), (evidence,)))

    assert indeterminate.status is DiagnosisStatus.INDETERMINATE
    assert narrowed.status is DiagnosisStatus.NARROWED


def test_pending_candidate_and_open_task_block_confirmation() -> None:
    evidence = _evidence()
    leader = _hypothesis("checkout", HypothesisStatus.LEADING, evidence.evidence_id)
    proposal = CandidateProposal(
        service="fraud",
        fault_type="delay",
        causal_claim="Fraud may be upstream.",
        evidence_ids=(evidence.evidence_id,),
        suggested_investigation="Compare fraud and checkout.",
    )
    task = InvestigationTask(
        task_id=str(uuid4()),
        task_kind=TaskKind.CHALLENGE,
        target_hypothesis_ids=(leader.hypothesis_id,),
        question="Seek an alternative.",
        shared_evidence_ids=(evidence.evidence_id,),
        discriminating_observations=("upstream onset",),
        available_modalities=(Modality.TRACES,),
    )

    with_proposal = evaluate_stop(_state((leader,), (evidence,), proposals=(proposal,)))
    with_task = evaluate_stop(_state((leader,), (evidence,), tasks=(task,)))

    assert with_proposal.status is not DiagnosisStatus.CONFIRMED
    assert with_task.status is not DiagnosisStatus.CONFIRMED


def test_worker_limitations_and_unresolved_questions_are_preserved_in_decision() -> None:
    evidence = _evidence(support=False)
    active = _hypothesis("checkout", HypothesisStatus.ACTIVE, evidence.evidence_id)
    result = InvestigationResult(
        task_id=str(uuid4()),
        evidence=(evidence,),
        conclusion="The candidates cannot be distinguished with current trace coverage.",
        cited_evidence_ids=(evidence.evidence_id,),
        unresolved_questions=("Need payment logs for the incident window.",),
        query_statuses={"logs-payment": QueryStatus.DATA_UNAVAILABLE},
    )

    decision = evaluate_stop(
        _state(
            (active,),
            (evidence,),
            results=(result,),
            query_statuses={"logs-payment": QueryStatus.DATA_UNAVAILABLE},
        )
    )

    assert "Need payment logs for the incident window." in decision.unresolved_questions
    assert any("DATA_UNAVAILABLE" in item for item in decision.limitations)


def test_observation_failure_requires_failed_observation_without_comparative_result() -> None:
    evidence = _evidence(support=False)
    active = _hypothesis("checkout", HypothesisStatus.ACTIVE, evidence.evidence_id)

    decision = evaluate_stop(
        _state(
            (active,),
            (evidence,),
            query_statuses={"traces-checkout": QueryStatus.TOOL_ERROR},
        )
    )

    assert decision.status is DiagnosisStatus.OBSERVATION_FAILURE


def test_supervisor_guard_stops_without_promoting_or_selecting_candidate() -> None:
    evidence = _evidence(support=False)
    active = _hypothesis("checkout", HypothesisStatus.ACTIVE, evidence.evidence_id)

    decision = evaluate_stop(
        _state(
            (active,),
            (evidence,),
            supervisor_steps=10,
            max_supervisor_steps=10,
        )
    )

    assert decision.status is DiagnosisStatus.INDETERMINATE
    assert decision.stop_reason == "MAX_SUPERVISOR_STEPS"
    assert decision.root_cause_hypothesis_id is None
    assert decision.remaining_hypothesis_ids == (active.hypothesis_id,)


def test_diagnosis_state_has_no_round_or_tool_budget_fields() -> None:
    assert "budget" not in DiagnosisState.model_fields
    assert "round" not in DiagnosisState.model_fields
