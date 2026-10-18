import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from poise_rca.domain import (
    CandidateProposal,
    CaseContext,
    DiagnosisStatus,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    InvestigationResult,
    Modality,
    QueryStatus,
)
from poise_rca.graph.state import DiagnosisState
from poise_rca.reporting import RunMetadata, build_report, write_report
from poise_rca.topology import build_observed_service_graph


def _state() -> DiagnosisState:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    checkout_evidence = EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.TRACES,
        service="checkout",
        observation="Checkout latency precedes frontend latency.",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.SUPPORT,
        source_reference="trace:checkout",
        attributes={"causal_support": True, "comparative": True},
        created_at=start,
    )
    fraud_evidence = EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.TRACES,
        service="fraud",
        observation="Fraud latency overlaps checkout latency.",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference="trace:fraud",
        created_at=start,
    )
    leader = Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service="checkout",
        fault_type="delay",
        causal_claim="Checkout delay initiated the incident.",
        supporting_evidence_ids=(checkout_evidence.evidence_id,),
        expected_observations=("Checkout rises first.",),
        status=HypothesisStatus.LEADING,
    )
    deferred = Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service="payment",
        fault_type="delay",
        causal_claim="Payment delay may contribute.",
        expected_observations=("Payment rises first.",),
        status=HypothesisStatus.DEFERRED,
    )
    fraud = Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service="fraud",
        fault_type="delay",
        causal_claim="Fraud delay may be upstream.",
        supporting_evidence_ids=(fraud_evidence.evidence_id,),
        expected_observations=("Fraud rises first.",),
        status=HypothesisStatus.ACTIVE,
    )
    proposal = CandidateProposal(
        service="fraud",
        fault_type="delay",
        causal_claim=fraud.causal_claim,
        evidence_ids=(fraud_evidence.evidence_id,),
        suggested_investigation="Compare fraud and checkout.",
    )
    result = InvestigationResult(
        task_id=str(uuid4()),
        evidence=(checkout_evidence, fraud_evidence),
        conclusion="Checkout is better supported, but fraud remains a viable upstream candidate.",
        cited_evidence_ids=(checkout_evidence.evidence_id, fraud_evidence.evidence_id),
        candidate_proposals=(proposal,),
        unresolved_questions=("Payment logs are unavailable.",),
        query_statuses={"payment-logs": QueryStatus.DATA_UNAVAILABLE},
    )
    return DiagnosisState(
        case=CaseContext(
            case_id="re2-ob-0000000000000001",
            incident_start=start,
            incident_end=start + timedelta(minutes=5),
            initial_symptom="Frontend latency increased.",
            available_modalities=frozenset({Modality.TRACES}),
        ),
        service_inventory=frozenset({"checkout", "payment", "fraud"}),
        topology=build_observed_service_graph((checkout_evidence, fraud_evidence)),
        evidence={
            checkout_evidence.evidence_id: checkout_evidence,
            fraud_evidence.evidence_id: fraud_evidence,
        },
        hypotheses={item.hypothesis_id: item for item in (leader, deferred, fraud)},
        tasks={},
        results=(result,),
        candidate_rejections=("ghost: absent from safe inventory",),
        query_statuses={"payment-logs": QueryStatus.DATA_UNAVAILABLE},
        conflicts=(),
        supervisor_steps=4,
    )


def _metadata() -> RunMetadata:
    return RunMetadata(
        run_id="run-001",
        code_sha="abcdef1",
        config_fingerprint="config-sha256",
        latency_ms=123.0,
        token_usage=42,
        model_cost=0.01,
        trace_id="trace-safe",
    )


def test_report_exposes_lifecycle_worker_conclusions_and_candidate_changes() -> None:
    state = _state()

    report = build_report(state, _metadata())

    assert report.status is DiagnosisStatus.NARROWED
    assert {item.status for item in report.candidates} == {
        HypothesisStatus.ACTIVE,
        HypothesisStatus.DEFERRED,
        HypothesisStatus.LEADING,
    }
    assert report.worker_conclusions[0].conclusion.startswith("Checkout is better supported")
    assert set(report.worker_conclusions[0].cited_evidence_ids) <= set(state.evidence)
    assert report.unresolved_questions == ("Payment logs are unavailable.",)
    assert report.candidate_promotions == ("fraud",)
    assert report.candidate_rejections == ("ghost: absent from safe inventory",)
    assert report.supervisor_steps == 4
    assert not hasattr(report.worker_conclusions[0], "outcome")


def test_report_claims_resolve_to_evidence_and_exclude_ground_truth() -> None:
    state = _state()
    report = build_report(state, _metadata())

    known_ids = set(state.evidence)
    assert all(set(claim.evidence_ids) <= known_ids for claim in report.confirmed_facts)
    assert all(
        citation.source_reference
        for claim in report.confirmed_facts
        for citation in claim.citations
    )
    serialized = report.model_dump_json().lower()
    assert "ground_truth" not in serialized
    assert "re2ob_checkout_delay" not in serialized
    assert "rounds" not in type(report).model_fields
    assert "calls" not in type(report).model_fields


def test_writer_atomically_emits_utf8_json(tmp_path: Path) -> None:
    report = build_report(_state(), _metadata())
    path = tmp_path / "run" / "diagnosis.json"

    write_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-001"
    assert payload["supervisor_steps"] == 4
    assert not list(path.parent.glob("*.tmp"))
