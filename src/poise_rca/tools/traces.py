"""Trace edge aggregation and normalized service-call evidence."""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from pydantic import BaseModel, ConfigDict

from poise_rca.data import TelemetryBundle
from poise_rca.domain import EvidencePolarity, EvidenceRecord, Modality, QueryStatus
from poise_rca.tools.base import (
    BaseTelemetryQuery,
    ToolQueryResult,
    elapsed_ms,
    find_column,
    timestamp_seconds,
)


class TraceQuery(BaseTelemetryQuery):
    """Query service-call edges originating from one service."""

    baseline_start: datetime | None = None
    baseline_end: datetime | None = None
    min_latency_relative_change: float = 0.5
    min_error_rate_increase: float = 0.1


class TraceEdgeSummary(BaseModel):
    """Aggregated service edge without raw spans or identifiers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_service: str
    target_service: str
    call_count: int
    error_count: int
    error_rate: float
    p95_latency_ms: float
    baseline_call_count: int | None = None
    baseline_error_rate: float | None = None
    baseline_p95_latency_ms: float | None = None
    error_anomaly: bool = False
    latency_anomaly: bool = False


def _result(
    query: TraceQuery,
    status: QueryStatus,
    records: tuple[TraceEdgeSummary, ...],
    started_at: datetime,
    error: str | None = None,
) -> ToolQueryResult[TraceEdgeSummary]:
    ended_at = datetime.now(UTC)
    return ToolQueryResult[TraceEdgeSummary](
        query_id=query.query_id,
        status=status,
        records=records,
        source_refs=(query.source_reference,),
        error=error,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=elapsed_ms(started_at, ended_at),
    )


def query_traces(
    bundle: TelemetryBundle,
    query: TraceQuery,
) -> ToolQueryResult[TraceEdgeSummary]:
    """Aggregate matching spans into observed service edges."""

    started_at = datetime.now(UTC)
    frame = bundle.traces
    if frame is None:
        return _result(
            query,
            QueryStatus.DATA_UNAVAILABLE,
            (),
            started_at,
            "trace modality is unavailable",
        )
    time_column = find_column(frame, ("timestamp", "time", "ts"))
    source_column = find_column(
        frame,
        ("service", "service_name", "source_service", "source", "caller"),
    )
    target_column = find_column(
        frame,
        ("peer_service", "target_service", "target", "callee", "destination"),
    )
    duration_column = find_column(
        frame,
        ("duration_ms", "latency_ms", "duration", "elapsed_ms"),
    )
    status_column = find_column(frame, ("status", "span_status", "code", "status_code"))
    if None in {time_column, source_column, duration_column, status_column}:
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "trace service, duration, status, or timestamp column is missing",
        )
    assert time_column is not None
    assert source_column is not None
    assert duration_column is not None
    assert status_column is not None

    working = frame.copy()
    if target_column is None:
        trace_column = find_column(working, ("trace_id", "traceid"))
        span_column = find_column(working, ("span_id", "spanid"))
        parent_column = find_column(working, ("parent_span_id", "parentspanid"))
        if None in {trace_column, span_column, parent_column}:
            return _result(
                query,
                QueryStatus.PARTIAL,
                (),
                started_at,
                "trace target column and parent-span linkage are both unavailable",
            )
        assert trace_column is not None
        assert span_column is not None
        assert parent_column is not None
        parent_services = {
            (str(row[trace_column]), str(row[span_column])): str(row[source_column])
            for _, row in working.iterrows()
        }
        working = working.assign(
            _source_service=[
                parent_services.get((str(trace), str(parent)), "")
                for trace, parent in zip(working[trace_column], working[parent_column], strict=True)
            ],
            _target_service=working[source_column].astype(str),
        )
        working = working[working["_source_service"] != ""]
        source_column = "_source_service"
        target_column = "_target_service"
    assert target_column is not None

    filtered = working.assign(
        _timestamp_seconds=timestamp_seconds(working[time_column]),
        _duration_ms=pd.to_numeric(working[duration_column], errors="coerce"),
    ).dropna(subset=["_timestamp_seconds", "_duration_ms"])
    filtered = filtered[
        filtered[source_column].astype(str).str.casefold() == query.service.casefold()
    ]
    normalized_status = filtered[status_column].astype(str).str.upper()
    filtered = filtered.assign(
        _is_error=~normalized_status.isin({"OK", "SUCCESS", "UNSET", "0", "200"})
    )
    incident = filtered[
        (filtered["_timestamp_seconds"] >= query.start_time.timestamp())
        & (filtered["_timestamp_seconds"] <= query.end_time.timestamp())
    ]
    if incident.empty:
        return _result(query, QueryStatus.SUCCESS_EMPTY, (), started_at)

    if (query.baseline_start is None) != (query.baseline_end is None):
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "both trace baseline bounds are required",
        )
    baseline: pd.DataFrame | None = None
    if query.baseline_start is not None and query.baseline_end is not None:
        baseline = filtered[
            (filtered["_timestamp_seconds"] >= query.baseline_start.timestamp())
            & (filtered["_timestamp_seconds"] < query.baseline_end.timestamp())
        ]
        if baseline.empty:
            return _result(
                query,
                QueryStatus.PARTIAL,
                (),
                started_at,
                "trace baseline window has no service-scoped samples",
            )

    summaries: list[TraceEdgeSummary] = []
    for (source, target), group in incident.groupby(
        [source_column, target_column], sort=True, dropna=False
    ):
        call_count = len(group)
        error_count = int(group["_is_error"].sum())
        baseline_group = (
            baseline[
                (baseline[source_column].astype(str) == str(source))
                & (baseline[target_column].astype(str) == str(target))
            ]
            if baseline is not None
            else None
        )
        baseline_call_count: int | None = None
        baseline_error_rate: float | None = None
        baseline_p95_latency_ms: float | None = None
        error_anomaly = False
        latency_anomaly = False
        incident_error_rate = error_count / call_count
        incident_p95 = float(group["_duration_ms"].quantile(0.95))
        if baseline_group is not None:
            baseline_call_count = len(baseline_group)
            if baseline_call_count > 0:
                baseline_error_rate = float(baseline_group["_is_error"].mean())
                baseline_p95_latency_ms = float(baseline_group["_duration_ms"].quantile(0.95))
                error_anomaly = (
                    incident_error_rate - baseline_error_rate >= query.min_error_rate_increase
                )
                latency_anomaly = (incident_p95 - baseline_p95_latency_ms) / max(
                    abs(baseline_p95_latency_ms), 1e-12
                ) >= query.min_latency_relative_change
        summaries.append(
            TraceEdgeSummary(
                source_service=str(source),
                target_service=str(target),
                call_count=call_count,
                error_count=error_count,
                error_rate=incident_error_rate,
                p95_latency_ms=incident_p95,
                baseline_call_count=baseline_call_count,
                baseline_error_rate=baseline_error_rate,
                baseline_p95_latency_ms=baseline_p95_latency_ms,
                error_anomaly=error_anomaly,
                latency_anomaly=latency_anomaly,
            )
        )
    return _result(query, QueryStatus.SUCCESS_WITH_DATA, tuple(summaries), started_at)


def normalize_trace_events(
    result: ToolQueryResult[TraceEdgeSummary],
) -> list[EvidenceRecord]:
    """Convert aggregate edges to neutral, provenance-bearing trace evidence."""

    polarity = (
        EvidencePolarity.NEUTRAL
        if result.status is QueryStatus.SUCCESS_WITH_DATA
        else EvidencePolarity.UNKNOWN
    )
    source_reference = result.source_refs[0] if result.source_refs else "traces:unavailable"
    evidence: list[EvidenceRecord] = []
    for index, record in enumerate(result.records):
        observation = (
            f"Observed {record.call_count} calls from {record.source_service} to "
            f"{record.target_service}; error rate {record.error_rate:.3g}, "
            f"p95 latency {record.p95_latency_ms:.6g} ms."
        )
        evidence.append(
            EvidenceRecord(
                evidence_id=str(
                    uuid5(NAMESPACE_URL, f"{result.query_id}:traces:{index}:{observation}")
                ),
                modality=Modality.TRACES,
                service=record.source_service,
                observation=observation,
                query_id=result.query_id,
                query_status=result.status,
                polarity=polarity,
                source_reference=f"{source_reference}#query={result.query_id}",
                attributes={
                    **record.model_dump(),
                    "signal_present": record.error_count > 0,
                    "anomaly": record.error_anomaly or record.latency_anomaly,
                    "anomaly_rule": "baseline_error_or_latency_increase",
                },
                created_at=result.ended_at,
            )
        )
    return evidence
