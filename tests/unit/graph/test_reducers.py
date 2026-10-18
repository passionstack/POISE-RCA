from datetime import UTC, datetime, timedelta
from uuid import uuid4

from poise_rca.domain import (
    CandidateProposal,
    CaseContext,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    TaskKind,
    TaskStatus,
)
from poise_rca.graph.reducers import merge_worker_results
from poise_rca.graph.state import DiagnosisState
from poise_rca.topology import build_observed_service_graph


def _evidence(
    service: str,
    observation: str,
    *,
    evidence_id: str | None = None,
    source: str = "trace:test",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id or str(uuid4()),
        modality=Modality.TRACES,
        service=service,
        observation=observation,
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference=source,
        created_at=datetime.now(UTC),
    )


def _hypothesis(service: str, evidence_id: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service=service,
        fault_type="delay",
        causal_claim=f"{service} delay initiated the incident.",
        origin_evidence_ids=(evidence_id,),
        expected_observations=(f"{service} anomaly precedes downstream symptoms",),
    )


def _state(
    *,
    evidence: tuple[EvidenceRecord, ...],
    hypotheses: tuple[Hypothesis, ...],
    task: InvestigationTask,
    inventory: frozenset[str] = frozenset({"checkout", "payment"}),
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
        service_inventory=inventory,
        topology=build_observed_service_graph(evidence),
        evidence={item.evidence_id: item for item in evidence},
        hypotheses={item.hypothesis_id: item for item in hypotheses},
        tasks={task.task_id: task},
        results=(),
        conflicts=(),
    )


def _task(hypothesis_id: str, evidence_id: str) -> InvestigationTask:
    return InvestigationTask(
        task_id=str(uuid4()),
        task_kind=TaskKind.CHALLENGE,
        target_hypothesis_ids=(hypothesis_id,),
        question="Can an upstream service better explain the anomaly?",
        shared_evidence_ids=(evidence_id,),
        discriminating_observations=("relative trace onset",),
        available_modalities=(Modality.TRACES,),
    )


def test_merge_adds_evidence_before_validating_conclusion_and_promoting_candidate() -> None:
    initial = _evidence("checkout", "Checkout latency rises.")
    hypothesis = _hypothesis("checkout", initial.evidence_id)
    task = _task(hypothesis.hypothesis_id, initial.evidence_id)
    discovered = _evidence("fraud", "Fraud latency rises before checkout.")
    proposal = CandidateProposal(
        service="fraud",
        fault_type="delay",
        causal_claim="fraud delay initiated the incident.",
        evidence_ids=(discovered.evidence_id,),
        suggested_investigation="Compare fraud and checkout onset.",
    )
    result = InvestigationResult(
        task_id=task.task_id,
        evidence=(initial, discovered),
        conclusion="Fraud is a credible upstream alternative.",
        cited_evidence_ids=(discovered.evidence_id,),
        candidate_proposals=(proposal,),
    )

    update = merge_worker_results(
        _state(evidence=(initial,), hypotheses=(hypothesis,), task=task),
        (result,),
    )

    promoted = next(
        item for item in update.hypotheses.values() if item.root_cause_service == "fraud"
    )
    assert promoted.status is HypothesisStatus.ACTIVE
    assert set(promoted.supporting_evidence_ids) <= set(update.evidence)
    assert update.pending_candidate_proposals
    assert update.tasks[task.task_id].status is TaskStatus.COMPLETED
    assert task.task_id not in update.invalid_conclusion_task_ids
    assert update.hypotheses[hypothesis.hypothesis_id].status is HypothesisStatus.ACTIVE


def test_dangling_conclusion_and_proposal_are_recorded_but_not_committed() -> None:
    initial = _evidence("checkout", "Checkout latency rises.")
    hypothesis = _hypothesis("checkout", initial.evidence_id)
    task = _task(hypothesis.hypothesis_id, initial.evidence_id)
    missing_id = str(uuid4())
    proposal = CandidateProposal(
        service="ghost",
        fault_type="delay",
        causal_claim="ghost delay initiated the incident.",
        evidence_ids=(missing_id,),
        suggested_investigation="Find the ghost service.",
    )
    raw = InvestigationResult.model_construct(
        task_id=task.task_id,
        evidence=(initial,),
        conclusion="Ghost is causal.",
        cited_evidence_ids=(missing_id,),
        candidate_proposals=(proposal,),
        unresolved_questions=(),
        query_signatures=(),
        query_id=None,
        status=None,
        error=None,
        started_at=None,
        finished_at=None,
    )

    update = merge_worker_results(
        _state(evidence=(initial,), hypotheses=(hypothesis,), task=task),
        (raw,),
    )

    assert update.results[-1].candidate_proposals == (proposal,)
    assert task.task_id in update.invalid_conclusion_task_ids
    assert update.pending_candidate_proposals == ()
    assert any("unknown evidence" in reason for reason in update.candidate_rejections)
    assert {item.root_cause_service for item in update.hypotheses.values()} == {"checkout"}


def test_duplicate_candidate_enriches_existing_hypothesis_without_new_id() -> None:
    initial = _evidence("checkout", "Checkout latency rises.")
    existing = _hypothesis("checkout", initial.evidence_id)
    task = _task(existing.hypothesis_id, initial.evidence_id)
    additional = _evidence("checkout", "Checkout trace errors precede the symptom.")
    proposal = CandidateProposal(
        service="checkout",
        fault_type="delay",
        causal_claim=existing.causal_claim,
        evidence_ids=(additional.evidence_id,),
        suggested_investigation="Compare trace and metric onset.",
    )
    result = InvestigationResult(
        task_id=task.task_id,
        evidence=(initial, additional),
        conclusion="New trace evidence strengthens the retained checkout candidate.",
        cited_evidence_ids=(additional.evidence_id,),
        candidate_proposals=(proposal,),
    )

    update = merge_worker_results(
        _state(evidence=(initial,), hypotheses=(existing,), task=task),
        (result,),
    )

    assert set(update.hypotheses) == {existing.hypothesis_id}
    enriched = update.hypotheses[existing.hypothesis_id]
    assert len(enriched.supporting_evidence_ids) == 1
    assert set(enriched.supporting_evidence_ids) <= set(update.evidence)


def test_invented_unconnected_service_is_rejected_with_reason() -> None:
    initial = _evidence("checkout", "Checkout latency rises.")
    hypothesis = _hypothesis("checkout", initial.evidence_id)
    task = _task(hypothesis.hypothesis_id, initial.evidence_id)
    proposal = CandidateProposal(
        service="ghost",
        fault_type="delay",
        causal_claim="ghost delay initiated the incident.",
        evidence_ids=(initial.evidence_id,),
        suggested_investigation="Inspect ghost telemetry.",
    )
    result = InvestigationResult(
        task_id=task.task_id,
        evidence=(initial,),
        conclusion="No alternative was established.",
        cited_evidence_ids=(initial.evidence_id,),
        candidate_proposals=(proposal,),
    )

    update = merge_worker_results(
        _state(evidence=(initial,), hypotheses=(hypothesis,), task=task),
        (result,),
    )

    assert update.pending_candidate_proposals == ()
    assert any(
        "safe inventory or observed telemetry" in item for item in update.candidate_rejections
    )


def test_parallel_duplicate_evidence_is_deduplicated_with_provenance() -> None:
    initial = _evidence("checkout", "Checkout latency rises.")
    hypothesis = _hypothesis("checkout", initial.evidence_id)
    task = _task(hypothesis.hypothesis_id, initial.evidence_id)
    first = _evidence("payment", "Payment errors rise.", source="trace:a")
    second = _evidence("payment", "Payment errors rise.", source="trace:b")
    results = tuple(
        InvestigationResult(
            task_id=task.task_id,
            evidence=(record,),
            conclusion="Payment remains a plausible contributor.",
            cited_evidence_ids=(record.evidence_id,),
        )
        for record in (first, second)
    )

    update = merge_worker_results(
        _state(evidence=(initial,), hypotheses=(hypothesis,), task=task),
        results,
    )

    payment_records = [item for item in update.evidence.values() if item.service == "payment"]
    assert len(payment_records) == 1
    assert payment_records[0].attributes["source_references"] == ["trace:a", "trace:b"]
    assert update.invalid_conclusion_task_ids == frozenset()
