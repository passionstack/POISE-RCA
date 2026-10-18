from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd

from poise_rca.data import TelemetryBundle
from poise_rca.domain import EvidencePolarity, QueryStatus
from poise_rca.tools.logs import LogQuery, normalize_log_events, query_logs


def _bundle(frame: pd.DataFrame) -> TelemetryBundle:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return TelemetryBundle(
        safe_case_id="re2-ob-0000000000000001",
        incident_start=start,
        incident_end=start + timedelta(minutes=5),
        initial_symptom="Elevated request failures.",
        entry_service="frontend",
        metrics=None,
        logs=frame,
        traces=None,
    )


def test_log_window_filtering_and_error_template_counts() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (start - timedelta(seconds=1)).timestamp(),
                (start + timedelta(seconds=10)).timestamp(),
                (start + timedelta(seconds=20)).timestamp(),
                (start + timedelta(seconds=30)).timestamp(),
            ],
            "service": ["checkout"] * 4,
            "severity": ["ERROR", "ERROR", "ERROR", "INFO"],
            "message": [
                "request 41 failed",
                "request 42 failed",
                "request 43 failed",
                "request 44 completed",
            ],
        }
    )
    query = LogQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        source_reference="logs.csv",
    )

    result = query_logs(_bundle(frame), query)
    evidence = normalize_log_events(result)

    assert result.status is QueryStatus.SUCCESS_WITH_DATA
    assert len(result.records) == 1
    assert result.records[0].count == 2
    assert result.records[0].template == "request <*> failed"
    assert evidence[0].polarity is EvidencePolarity.NEUTRAL
    assert evidence[0].attributes["anomaly"] is False
    assert evidence[0].attributes["signal_present"] is True
    assert evidence[0].source_reference.startswith("logs.csv#query=")
    assert "request 42" not in evidence[0].observation


def test_log_query_can_succeed_with_no_matching_data() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [start.timestamp()],
            "service": ["checkout"],
            "severity": ["INFO"],
            "message": ["request completed"],
        }
    )
    query = LogQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        source_reference="logs.csv",
    )

    result = query_logs(_bundle(frame), query)

    assert result.status is QueryStatus.SUCCESS_EMPTY
    assert result.records == ()


def test_rcaeval_container_name_scopes_log_query() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [start.timestamp(), start.timestamp()],
            "container_name": ["checkout", "payment"],
            "severity": ["ERROR", "ERROR"],
            "message": ["checkout failed", "payment failed"],
        }
    )
    query = LogQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        source_reference="case:logs:sha256:abc",
    )

    result = query_logs(_bundle(frame), query)

    assert result.status is QueryStatus.SUCCESS_WITH_DATA
    assert len(result.records) == 1
    assert result.records[0].service == "checkout"
    assert "payment" not in result.records[0].template


def test_log_query_without_service_column_is_partial_not_misattributed() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [start.timestamp()],
            "severity": ["ERROR"],
            "message": ["unscoped failure"],
        }
    )
    query = LogQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        source_reference="case:logs:sha256:abc",
    )

    result = query_logs(_bundle(frame), query)

    assert result.status is QueryStatus.PARTIAL
    assert result.records == ()


def test_log_rate_increase_against_baseline_is_anomalous() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (start - timedelta(seconds=30)).timestamp(),
                (start + timedelta(seconds=10)).timestamp(),
                (start + timedelta(seconds=20)).timestamp(),
            ],
            "service": ["checkout"] * 3,
            "severity": ["ERROR"] * 3,
            "message": ["request failed"] * 3,
        }
    )
    query = LogQuery(
        query_id=str(uuid4()),
        service="checkout",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        baseline_start=start - timedelta(minutes=1),
        baseline_end=start,
        source_reference="logs.csv",
    )

    evidence = normalize_log_events(query_logs(_bundle(frame), query))

    assert evidence[0].attributes["baseline_count"] == 1
    assert evidence[0].attributes["incident_count"] == 2
    assert evidence[0].attributes["anomaly"] is True
