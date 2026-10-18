from uuid import uuid4

import pytest
from pydantic import ValidationError

from poise_rca.domain import AnomalyFinding, Modality, QueryStatus, StaticAnalysisResult


def _finding(**overrides: object) -> AnomalyFinding:
    values: dict[str, object] = {
        "finding_id": str(uuid4()),
        "service": "checkout",
        "modality": Modality.METRICS,
        "signal": "checkout_cpu_usage",
        "observation": "CPU usage rises at incident start.",
        "source_reference": "metric:checkout_cpu_usage",
        "analyzer": "rcaeval-mmbaro",
        "score": 0.91,
        "rank": 1,
    }
    values.update(overrides)
    return AnomalyFinding.model_validate(values)


def test_anomaly_finding_retains_analyzer_rank_score_and_source() -> None:
    finding = _finding()

    assert finding.analyzer == "rcaeval-mmbaro"
    assert finding.rank == 1
    assert finding.score == 0.91
    assert finding.source_reference == "metric:checkout_cpu_usage"


def test_static_analysis_distinguishes_success_empty_unavailable_and_failure() -> None:
    finding = _finding()
    result = StaticAnalysisResult(
        findings=(finding,),
        statuses={
            Modality.METRICS: QueryStatus.SUCCESS_WITH_DATA,
            Modality.LOGS: QueryStatus.SUCCESS_EMPTY,
            Modality.TRACES: QueryStatus.DATA_UNAVAILABLE,
        },
        errors={Modality.TRACES: "trace source unavailable"},
    )

    assert result.findings == (finding,)
    assert result.statuses[Modality.LOGS] is QueryStatus.SUCCESS_EMPTY
    assert result.statuses[Modality.TRACES] is QueryStatus.DATA_UNAVAILABLE
    assert result.errors == {Modality.TRACES: "trace source unavailable"}


@pytest.mark.parametrize("field", ["service", "signal", "source_reference", "analyzer"])
def test_finding_rejects_blank_service_signal_or_source(field: str) -> None:
    with pytest.raises(ValidationError):
        _finding(**{field: "   "})
