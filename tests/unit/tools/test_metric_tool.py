from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest

from poise_rca.data import TelemetryBundle
from poise_rca.domain import QueryStatus
from poise_rca.tools.metrics import MetricQuery, normalize_metric_events, query_metrics


def _bundle(frame: pd.DataFrame) -> TelemetryBundle:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return TelemetryBundle(
        safe_case_id="re2-ob-0000000000000001",
        incident_start=start,
        incident_end=start + timedelta(minutes=5),
        initial_symptom="Elevated latency.",
        entry_service="frontend",
        metrics=frame,
        logs=None,
        traces=None,
    )


def test_metric_baseline_versus_incident_anomaly_summary() -> None:
    incident = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (incident - timedelta(minutes=2)).timestamp(),
                (incident - timedelta(minutes=1)).timestamp(),
                (incident + timedelta(minutes=1)).timestamp(),
                (incident + timedelta(minutes=2)).timestamp(),
            ],
            "service": ["checkout"] * 4,
            "metric": ["latency_ms"] * 4,
            "value": [10.0, 10.0, 30.0, 32.0],
        }
    )
    query = MetricQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=incident,
        end_time=incident + timedelta(minutes=3),
        baseline_start=incident - timedelta(minutes=3),
        baseline_end=incident,
        source_reference="metrics.csv",
        min_relative_change=0.5,
    )

    result = query_metrics(_bundle(frame), query)
    evidence = normalize_metric_events(result)

    assert result.status is QueryStatus.SUCCESS_WITH_DATA
    assert result.records[0].baseline_mean == pytest.approx(10.0)
    assert result.records[0].incident_mean == pytest.approx(31.0)
    assert result.records[0].relative_change == pytest.approx(2.1)
    assert "31" in evidence[0].observation
    assert evidence[0].attributes["anomaly"] is True
    assert evidence[0].source_reference.startswith("metrics.csv#query=")


def test_metric_query_can_succeed_with_no_anomaly() -> None:
    incident = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (incident - timedelta(minutes=1)).timestamp(),
                (incident + timedelta(minutes=1)).timestamp(),
            ],
            "service": ["checkout", "checkout"],
            "metric": ["cpu", "cpu"],
            "value": [0.4, 0.41],
        }
    )
    query = MetricQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=incident,
        end_time=incident + timedelta(minutes=2),
        baseline_start=incident - timedelta(minutes=2),
        baseline_end=incident,
        source_reference="metrics.csv",
        min_relative_change=0.5,
    )

    result = query_metrics(_bundle(frame), query)

    assert result.status is QueryStatus.SUCCESS_EMPTY


def test_metric_query_reports_partial_when_baseline_is_missing() -> None:
    incident = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [(incident + timedelta(minutes=1)).timestamp()],
            "service": ["checkout"],
            "metric": ["latency_ms"],
            "value": [30.0],
        }
    )
    query = MetricQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=incident,
        end_time=incident + timedelta(minutes=2),
        baseline_start=incident - timedelta(minutes=2),
        baseline_end=incident,
        source_reference="metrics.csv",
    )

    result = query_metrics(_bundle(frame), query)

    assert result.status is QueryStatus.PARTIAL
    assert result.error == "metric baseline window has no usable samples"


def test_long_metrics_without_service_scope_are_partial() -> None:
    incident = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (incident - timedelta(minutes=1)).timestamp(),
                (incident + timedelta(minutes=1)).timestamp(),
            ],
            "metric": ["latency_ms", "latency_ms"],
            "value": [10.0, 30.0],
        }
    )
    query = MetricQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=incident,
        end_time=incident + timedelta(minutes=2),
        baseline_start=incident - timedelta(minutes=2),
        baseline_end=incident,
        source_reference="metrics.csv",
    )

    result = query_metrics(_bundle(frame), query)

    assert result.status is QueryStatus.PARTIAL
    assert result.error == "metric service column is missing; rows cannot be safely scoped"


def test_wide_metrics_without_requested_service_are_not_misattributed() -> None:
    incident = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (incident - timedelta(minutes=1)).timestamp(),
                (incident + timedelta(minutes=1)).timestamp(),
            ],
            "payment_cpu": [0.1, 0.9],
        }
    )
    query = MetricQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=incident,
        end_time=incident + timedelta(minutes=2),
        baseline_start=incident - timedelta(minutes=2),
        baseline_end=incident,
        source_reference="metrics.csv",
    )

    result = query_metrics(_bundle(frame), query)

    assert result.status is QueryStatus.SUCCESS_EMPTY
    assert result.records == ()
