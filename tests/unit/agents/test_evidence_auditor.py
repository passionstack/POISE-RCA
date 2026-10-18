from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from poise_rca.agents.evidence_auditor import AuditResult, ClaimCheck, audit_claims
from poise_rca.domain import (
    CaseContext,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    Modality,
    QueryStatus,
)
from poise_rca.graph.state import DiagnosisState
from poise_rca.topology import build_observed_service_graph


class FakeAuditModel:
    def __init__(self, result: AuditResult) -> None:
        self.result = result

    def invoke(self, input: dict[str, object]) -> object:
        del input
        return self.result


def _state() -> tuple[DiagnosisState, EvidenceRecord]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    evidence = EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.METRICS,
        service="checkout",
        observation="checkout latency increased",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference="metrics.csv#query=seed",
        created_at=start,
    )
    hypothesis = Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service="checkout",
        fault_type="delay",
        causal_claim="checkout delay may explain frontend latency",
        origin_evidence_ids=(evidence.evidence_id,),
        missing_evidence=("independent trace evidence",),
    )
    state = DiagnosisState(
        case=CaseContext(
            case_id="re2-ob-0000000000000001",
            incident_start=start,
            incident_end=start + timedelta(minutes=5),
            initial_symptom="Frontend latency increased.",
            available_modalities=frozenset({Modality.METRICS}),
        ),
        topology=build_observed_service_graph([evidence]),
        evidence={evidence.evidence_id: evidence},
        hypotheses={hypothesis.hypothesis_id: hypothesis},
        tasks={},
        results=(),
        conflicts=(),
    )
    return state, evidence


def test_auditor_cannot_reference_invented_evidence() -> None:
    state, _ = _state()
    invented_id = str(uuid4())
    model = FakeAuditModel(
        AuditResult(
            claims=(
                ClaimCheck(
                    claim="invented fact",
                    evidence_ids=(invented_id,),
                    grounded=True,
                ),
            )
        )
    )

    with pytest.raises(ValueError, match="unknown evidence ID"):
        audit_claims(state, model)


def test_auditor_preserves_known_claim_links() -> None:
    state, evidence = _state()
    expected = AuditResult(
        claims=(
            ClaimCheck(
                claim=evidence.observation,
                evidence_ids=(evidence.evidence_id,),
                grounded=True,
            ),
        ),
        unresolved_competitor_warnings=("OTHER/UNKNOWN remains unresolved",),
    )

    assert audit_claims(state, FakeAuditModel(expected)) == expected
