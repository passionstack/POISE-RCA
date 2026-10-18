from datetime import UTC, datetime, timedelta
from uuid import uuid4

from poise_rca.agents import (
    HypothesisBatch,
    HypothesisDraft,
    HypothesisPromptInput,
    generate_initial_hypotheses,
)
from poise_rca.analysis import finding_to_evidence
from poise_rca.domain import (
    AnomalyFinding,
    CaseContext,
    CausalRole,
    EvidenceRecord,
    HypothesisStatus,
    Modality,
    QueryStatus,
    StaticAnalysisResult,
)
from poise_rca.topology import build_observed_service_graph


class FakeStructuredModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.inputs: list[HypothesisPromptInput] = []

    def invoke(self, input: HypothesisPromptInput) -> object:
        self.inputs.append(input)
        return self.responses.pop(0)


def _context() -> CaseContext:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return CaseContext(
        case_id="re2-ob-0000000000000001",
        incident_start=start,
        incident_end=start + timedelta(minutes=5),
        initial_symptom="Elevated frontend latency.",
        entry_service="frontend",
        available_modalities=frozenset({Modality.METRICS, Modality.TRACES}),
    )


def _analysis(*services: str) -> tuple[StaticAnalysisResult, tuple[EvidenceRecord, ...]]:
    findings = tuple(
        AnomalyFinding(
            finding_id=str(uuid4()),
            service=service,
            modality=Modality.METRICS,
            signal=f"{service}_latency",
            observation=f"{service} latency changed near incident onset.",
            source_reference=f"metric:{service}_latency",
            analyzer="fake-static",
            score=1.0 / rank,
            rank=rank,
        )
        for rank, service in enumerate(services, start=1)
    )
    analysis = StaticAnalysisResult(
        findings=findings,
        statuses={
            Modality.METRICS: QueryStatus.SUCCESS_WITH_DATA,
            Modality.LOGS: QueryStatus.DATA_UNAVAILABLE,
            Modality.TRACES: QueryStatus.SUCCESS_EMPTY,
        },
    )
    evidence = tuple(finding_to_evidence(item, datetime.now(UTC)) for item in findings)
    return analysis, evidence


def _draft(
    service: str,
    evidence_ids: tuple[str, ...],
    *,
    role: CausalRole = CausalRole.UNKNOWN,
) -> HypothesisDraft:
    return HypothesisDraft(
        root_cause_service=service,
        fault_type="delay",
        causal_claim=f"{service} delay explains the symptom",
        causal_role=role,
        origin_evidence_ids=evidence_ids,
        expected_observations=(f"independent evidence for {service}",),
    )


def test_four_findings_can_produce_more_than_two_active_hypotheses() -> None:
    services = ("checkout", "payment", "catalog", "frontend")
    analysis, evidence = _analysis(*services)
    model = FakeStructuredModel(
        [
            HypothesisBatch(
                hypotheses=tuple(
                    _draft(service, (record.evidence_id,))
                    for service, record in zip(services, evidence, strict=True)
                )
            )
        ]
    )

    hypotheses = generate_initial_hypotheses(
        _context(),
        build_observed_service_graph(evidence),
        frozenset(services),
        analysis,
        evidence,
        model,
    )

    concrete = [item for item in hypotheses.values() if item.root_cause_service != "OTHER/UNKNOWN"]
    assert len(concrete) == 4
    assert all(item.status is HypothesisStatus.ACTIVE for item in concrete)
    assert not any(item.status is HypothesisStatus.LEADING for item in concrete)


def test_hypothesis_can_combine_multiple_finding_evidence_ids() -> None:
    analysis, evidence = _analysis("checkout", "checkout")
    combined_ids = tuple(record.evidence_id for record in evidence)
    model = FakeStructuredModel(
        [
            HypothesisBatch(
                hypotheses=(_draft("checkout", combined_ids, role=CausalRole.INITIATING_CAUSE),)
            )
        ]
    )

    hypotheses = generate_initial_hypotheses(
        _context(),
        build_observed_service_graph(evidence),
        frozenset({"checkout"}),
        analysis,
        evidence,
        model,
    )

    checkout = next(item for item in hypotheses.values() if item.root_cause_service == "checkout")
    assert checkout.origin_evidence_ids == combined_ids
    assert checkout.causal_role is CausalRole.INITIATING_CAUSE


def test_prompt_exposes_rank_and_missing_modality_without_promoting_a_leader() -> None:
    analysis, evidence = _analysis("checkout")
    model = FakeStructuredModel(
        [HypothesisBatch(hypotheses=(_draft("checkout", (evidence[0].evidence_id,)),))]
    )

    hypotheses = generate_initial_hypotheses(
        _context(),
        build_observed_service_graph(evidence),
        frozenset({"checkout"}),
        analysis,
        evidence,
        model,
    )

    assert model.inputs[0].findings[0].rank == 1
    assert model.inputs[0].analysis_statuses[Modality.LOGS] is QueryStatus.DATA_UNAVAILABLE
    assert all(item.status is not HypothesisStatus.LEADING for item in hypotheses.values())


def test_invented_services_and_evidence_ids_are_rejected() -> None:
    analysis, evidence = _analysis("checkout")
    known_id = evidence[0].evidence_id
    model = FakeStructuredModel(
        [
            HypothesisBatch(
                hypotheses=(
                    _draft("invented-service", (known_id,)),
                    _draft("checkout", (str(uuid4()),)),
                    _draft("checkout", (known_id,)),
                )
            )
        ]
    )

    hypotheses = generate_initial_hypotheses(
        _context(),
        build_observed_service_graph(evidence),
        frozenset({"checkout"}),
        analysis,
        evidence,
        model,
    )

    assert {item.root_cause_service for item in hypotheses.values()} == {
        "checkout",
        "OTHER/UNKNOWN",
    }


def test_empty_findings_retain_only_other_unknown() -> None:
    analysis, evidence = _analysis()
    model = FakeStructuredModel([HypothesisBatch(hypotheses=())])

    hypotheses = generate_initial_hypotheses(
        _context(),
        build_observed_service_graph(evidence),
        frozenset(),
        analysis,
        evidence,
        model,
    )

    assert [item.root_cause_service for item in hypotheses.values()] == ["OTHER/UNKNOWN"]


def test_schema_failure_gets_one_repair_request() -> None:
    analysis, evidence = _analysis("checkout")
    repaired = HypothesisBatch(hypotheses=(_draft("checkout", (evidence[0].evidence_id,)),))
    model = FakeStructuredModel([{"wrong": "shape"}, repaired])

    generate_initial_hypotheses(
        _context(),
        build_observed_service_graph(evidence),
        frozenset({"checkout"}),
        analysis,
        evidence,
        model,
    )

    assert len(model.inputs) == 2
    assert model.inputs[1].repair_error is not None
