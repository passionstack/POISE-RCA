from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest

from poise_rca.data import TelemetryBundle
from poise_rca.domain import QueryStatus
from poise_rca.tools.traces import TraceQuery, normalize_trace_events, query_traces


def _bundle(frame: pd.DataFrame) -> TelemetryBundle:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return TelemetryBundle(
        safe_case_id="re2-ob-0000000000000001",
        incident_start=start,
        incident_end=start + timedelta(minutes=5),
        initial_symptom="Elevated request failures.",
        entry_service="frontend",
        metrics=None,
        logs=None,
        traces=frame,
    )


def test_trace_service_edges_error_spans_and_latency_summary() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (start + timedelta(seconds=1)).timestamp(),
                (start + timedelta(seconds=2)).timestamp(),
                (start + timedelta(seconds=3)).timestamp(),
            ],
            "service": ["frontend", "frontend", "checkout"],
            "peer_service": ["checkout", "checkout", "payment"],
            "duration_ms": [100.0, 900.0, 400.0],
            "status": ["OK", "ERROR", "ERROR"],
        }
    )
    query = TraceQuery(
        query_id=str(uuid4()),
        service="frontend",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        source_reference="traces.csv",
    )

    result = query_traces(_bundle(frame), query)
    evidence = normalize_trace_events(result)

    assert result.status is QueryStatus.SUCCESS_WITH_DATA
    assert len(result.records) == 1
    assert result.records[0].source_service == "frontend"
    assert result.records[0].target_service == "checkout"
    assert result.records[0].call_count == 2
    assert result.records[0].error_count == 1
    assert result.records[0].p95_latency_ms == pytest.approx(860.0)
    assert evidence[0].attributes["target_service"] == "checkout"
    assert evidence[0].attributes["anomaly"] is False
    assert evidence[0].attributes["signal_present"] is True


def test_trace_query_can_succeed_with_no_matching_edges() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [start.timestamp()],
            "service": ["checkout"],
            "peer_service": ["payment"],
            "duration_ms": [50.0],
            "status": ["OK"],
        }
    )
    query = TraceQuery(
        query_id=str(uuid4()),
        service="frontend",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        source_reference="traces.csv",
    )

    result = query_traces(_bundle(frame), query)

    assert result.status is QueryStatus.SUCCESS_EMPTY


def test_error_free_trace_edge_remains_non_anomalous() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [start.timestamp()],
            "service": ["frontend"],
            "peer_service": ["checkout"],
            "duration_ms": [50.0],
            "status": ["OK"],
        }
    )
    query = TraceQuery(
        query_id=str(uuid4()),
        service="frontend",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        source_reference="traces.csv",
    )

    evidence = normalize_trace_events(query_traces(_bundle(frame), query))

    assert evidence[0].attributes["anomaly"] is False


def test_span_oriented_rcaeval_traces_reconstruct_parent_child_edges() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [start.timestamp(), start.timestamp() + 0.1],
            "trace_id": ["trace-1", "trace-1"],
            "span_id": ["parent", "child"],
            "parent_span_id": ["", "parent"],
            "service_name": ["frontend", "checkout"],
            "duration_ms": [800.0, 700.0],
            "status_code": ["ERROR", "ERROR"],
        }
    )
    query = TraceQuery(
        query_id=str(uuid4()),
        service="frontend",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        source_reference="case:traces:sha256:abc",
    )

    result = query_traces(_bundle(frame), query)

    assert result.status is QueryStatus.SUCCESS_WITH_DATA
    assert result.records[0].source_service == "frontend"
    assert result.records[0].target_service == "checkout"


def test_trace_latency_increase_against_baseline_is_anomalous() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (start - timedelta(seconds=30)).timestamp(),
                (start + timedelta(seconds=10)).timestamp(),
            ],
            "service": ["frontend", "frontend"],
            "peer_service": ["checkout", "checkout"],
            "duration_ms": [100.0, 500.0],
            "status": ["OK", "OK"],
        }
    )
    query = TraceQuery(
        query_id=str(uuid4()),
        service="frontend",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        baseline_start=start - timedelta(minutes=1),
        baseline_end=start,
        source_reference="traces.csv",
    )

    evidence = normalize_trace_events(query_traces(_bundle(frame), query))

    assert evidence[0].attributes["baseline_p95_latency_ms"] == pytest.approx(100.0)
    assert evidence[0].attributes["latency_anomaly"] is True
    assert evidence[0].attributes["anomaly"] is True
