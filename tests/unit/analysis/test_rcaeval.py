from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from poise_rca.analysis import RCAEvalMMBaroAnalyzer
from poise_rca.data import TelemetryBundle
from poise_rca.domain import Modality, QueryStatus


def _bundle() -> TelemetryBundle:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    timestamps = [
        start.timestamp(),
        (start + timedelta(seconds=5)).timestamp(),
        (start + timedelta(seconds=20)).timestamp(),
    ]
    return TelemetryBundle(
        safe_case_id="re2-ob-0000000000000001",
        incident_start=start,
        incident_end=start + timedelta(minutes=5),
        initial_symptom="Elevated request latency.",
        entry_service="frontend",
        metrics=pd.DataFrame(
            {
                "timestamp": timestamps,
                "checkout_cpu_usage": [0.1, 0.2, 0.9],
                "unknownservice_cpu_usage": [0.2, 0.2, 0.8],
            }
        ),
        logs=pd.DataFrame(
            {
                "timestamp": timestamps,
                "service": ["checkout", "checkout", "checkout"],
                "message": ["request 41 failed", "request 42 failed", "request completed"],
            }
        ),
        traces=pd.DataFrame(
            {
                "timestamp": timestamps,
                "service": ["frontend", "frontend", "frontend"],
                "peer_service": ["checkout", "checkout", "checkout"],
                "duration_ms": [50.0, 70.0, 500.0],
                "status": ["OK", "ERROR", "ERROR"],
            }
        ),
    )


def test_mmbaro_prepares_multisource_frames_and_maps_ranked_indicators() -> None:
    captured: dict[str, Any] = {}

    def backend(
        data: dict[str, pd.DataFrame], inject_time: float, *, dataset: str
    ) -> dict[str, object]:
        captured.update({"data": data, "inject_time": inject_time, "dataset": dataset})
        log_indicator = next(column for column in data["logts"] if column != "time")
        error_indicator = next(column for column in data["tracets_err"] if column != "time")
        latency_indicator = next(column for column in data["tracets_lat"] if column != "time")
        ranks = [
            "checkout_cpu_usage",
            log_indicator,
            error_indicator,
            latency_indicator,
            "unknownservice_cpu_usage",
        ]
        return {
            "ranks": ranks,
            "scores": {indicator: 1.0 / rank for rank, indicator in enumerate(ranks, start=1)},
        }

    bundle = _bundle()
    result = RCAEvalMMBaroAnalyzer(backend=backend).analyze(
        bundle.agent_view(), bundle, frozenset({"checkout", "frontend"})
    )

    prepared = captured["data"]
    assert captured["dataset"] == "re2-ob"
    expected_injection = bundle.incident_start + (bundle.incident_end - bundle.incident_start) / 2
    assert captured["inject_time"] == expected_injection.timestamp()
    assert list(prepared["metric"].columns)[0] == "time"
    assert len(prepared["logts"]) == 2
    assert prepared["logts"].iloc[0].drop(labels="time").sum() == 2
    assert prepared["tracets_err"].iloc[0].drop(labels="time").sum() == 1
    assert prepared["tracets_lat"].iloc[1].drop(labels="time").max() == 500.0

    assert [finding.rank for finding in result.findings] == [1, 2, 3, 4]
    assert [finding.modality for finding in result.findings] == [
        Modality.METRICS,
        Modality.LOGS,
        Modality.TRACES,
        Modality.TRACES,
    ]
    assert [finding.score for finding in result.findings] == [1.0, 0.5, 1 / 3, 0.25]
    assert result.unmapped_indicators == ("unknownservice_cpu_usage",)
    assert set(result.statuses.values()) == {QueryStatus.SUCCESS_WITH_DATA}


def test_mmbaro_preserves_rank_without_fabricating_scores() -> None:
    bundle = _bundle()

    def backend(
        data: dict[str, pd.DataFrame], inject_time: float, *, dataset: str
    ) -> dict[str, object]:
        del data, inject_time, dataset
        return {"ranks": ["checkout_cpu_usage"]}

    result = RCAEvalMMBaroAnalyzer(backend=backend).analyze(
        bundle.agent_view(), bundle, frozenset({"checkout", "frontend"})
    )

    assert result.findings[0].rank == 1
    assert result.findings[0].score is None


def test_mmbaro_marks_absent_modalities_unavailable() -> None:
    bundle = _bundle()
    metrics_only = TelemetryBundle(
        safe_case_id=bundle.safe_case_id,
        incident_start=bundle.incident_start,
        incident_end=bundle.incident_end,
        initial_symptom=bundle.initial_symptom,
        entry_service=bundle.entry_service,
        metrics=bundle.metrics,
        logs=None,
        traces=None,
    )

    result = RCAEvalMMBaroAnalyzer(
        backend=lambda data, inject_time, dataset: {"ranks": ["checkout_cpu_usage"]}
    ).analyze(metrics_only.agent_view(), metrics_only, frozenset({"checkout"}))

    assert result.statuses == {
        Modality.METRICS: QueryStatus.SUCCESS_WITH_DATA,
        Modality.LOGS: QueryStatus.DATA_UNAVAILABLE,
        Modality.TRACES: QueryStatus.DATA_UNAVAILABLE,
    }
