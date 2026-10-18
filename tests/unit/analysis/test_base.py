from datetime import UTC, datetime, timedelta
from uuid import uuid4

from poise_rca.analysis import StaticAnalyzer, finding_to_evidence
from poise_rca.data import TelemetryBundle
from poise_rca.domain import (
    AnomalyFinding,
    CaseContext,
    EvidencePolarity,
    Modality,
    QueryStatus,
    StaticAnalysisResult,
)


class FakeAnalyzer:
    def analyze(
        self,
        case: CaseContext,
        telemetry: TelemetryBundle,
        service_inventory: frozenset[str],
    ) -> StaticAnalysisResult:
        del case, telemetry
        return StaticAnalysisResult(
            findings=tuple(
                AnomalyFinding(
                    finding_id=str(uuid4()),
                    service=service,
                    modality=Modality.METRICS,
                    signal=f"{service}_cpu_usage",
                    observation=f"{service} CPU usage changed at incident start.",
                    source_reference=f"metric:{service}_cpu_usage",
                    analyzer="fake-static",
                    score=0.8,
                    rank=rank,
                )
                for rank, service in enumerate(sorted(service_inventory), start=1)
            ),
            statuses={
                Modality.METRICS: QueryStatus.SUCCESS_WITH_DATA,
                Modality.LOGS: QueryStatus.SUCCESS_EMPTY,
                Modality.TRACES: QueryStatus.DATA_UNAVAILABLE,
            },
        )


def _bundle() -> TelemetryBundle:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return TelemetryBundle(
        safe_case_id="re2-ob-0000000000000001",
        incident_start=start,
        incident_end=start + timedelta(minutes=5),
        initial_symptom="Elevated latency.",
        entry_service="frontend",
        metrics=None,
        logs=None,
        traces=None,
    )


def test_injected_analyzer_can_return_multiple_findings() -> None:
    analyzer: StaticAnalyzer = FakeAnalyzer()
    bundle = _bundle()

    result = analyzer.analyze(bundle.agent_view(), bundle, frozenset({"checkout", "payment"}))

    assert [finding.service for finding in result.findings] == ["checkout", "payment"]
    assert result.statuses[Modality.TRACES] is QueryStatus.DATA_UNAVAILABLE


def test_finding_conversion_preserves_provenance() -> None:
    finding = (
        FakeAnalyzer()
        .analyze(_bundle().agent_view(), _bundle(), frozenset({"checkout"}))
        .findings[0]
    )
    created_at = datetime(2025, 1, 1, tzinfo=UTC)

    evidence = finding_to_evidence(finding, created_at)

    assert evidence.service == "checkout"
    assert evidence.source_reference == "metric:checkout_cpu_usage"
    assert evidence.query_status is QueryStatus.SUCCESS_WITH_DATA
    assert evidence.polarity is EvidencePolarity.NEUTRAL
    assert evidence.attributes == {
        "analyzer": "fake-static",
        "finding_id": finding.finding_id,
        "rank": 1,
        "score": 0.8,
        "signal": "checkout_cpu_usage",
        "stage": "initialization",
    }


def test_unavailable_modality_produces_no_anomaly_evidence() -> None:
    result = FakeAnalyzer().analyze(_bundle().agent_view(), _bundle(), frozenset())

    evidence = tuple(finding_to_evidence(item, datetime.now(UTC)) for item in result.findings)

    assert result.statuses[Modality.TRACES] is QueryStatus.DATA_UNAVAILABLE
    assert evidence == ()
