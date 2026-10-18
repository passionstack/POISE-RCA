"""Windowed log querying and deterministic error-template normalization."""

import re
from datetime import UTC, datetime
from typing import Any, cast
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

_ERROR_WORDS = re.compile(r"\b(error|fail(?:ed|ure)?|exception|timeout|deadline)\b", re.I)
_VARIABLE_TOKEN = re.compile(
    r"\b(?:[0-9a-f]{8}-[0-9a-f-]{27,}|0x[0-9a-f]+|\d+(?:\.\d+)?)\b",
    re.I,
)


class LogQuery(BaseTelemetryQuery):
    """Query for error-like log templates in one service and time window."""

    baseline_start: datetime | None = None
    baseline_end: datetime | None = None


class LogEventSummary(BaseModel):
    """Aggregated log event; no raw log row or message is exposed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: str
    template: str
    count: int
    baseline_count: int | None = None
    incident_count: int
    rate_increase: float | None = None
    anomalous: bool = False
    severity: str
    first_timestamp: float
    last_timestamp: float


def _result(
    query: LogQuery,
    status: QueryStatus,
    records: tuple[LogEventSummary, ...],
    started_at: datetime,
    error: str | None = None,
) -> ToolQueryResult[LogEventSummary]:
    ended_at = datetime.now(UTC)
    return ToolQueryResult[LogEventSummary](
        query_id=query.query_id,
        status=status,
        records=records,
        source_refs=(query.source_reference,),
        error=error,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=elapsed_ms(started_at, ended_at),
    )


def query_logs(bundle: TelemetryBundle, query: LogQuery) -> ToolQueryResult[LogEventSummary]:
    """Filter logs and aggregate error templates without returning raw rows."""

    started_at = datetime.now(UTC)
    frame = bundle.logs
    if frame is None:
        return _result(
            query,
            QueryStatus.DATA_UNAVAILABLE,
            (),
            started_at,
            "log modality is unavailable",
        )

    time_column = find_column(frame, ("timestamp", "time", "ts"))
    message_column = find_column(frame, ("message", "log", "body", "content"))
    service_column = find_column(
        frame,
        (
            "service",
            "service_name",
            "servicename",
            "container_name",
            "containername",
            "pod",
            "pod_name",
            "component",
        ),
    )
    severity_column = find_column(frame, ("severity", "level", "log_level"))
    if time_column is None or message_column is None:
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "log timestamp or message column is missing",
        )
    if service_column is None:
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "log service column is missing; rows cannot be safely scoped",
        )

    filtered = frame.assign(_timestamp_seconds=timestamp_seconds(frame[time_column]))
    filtered = filtered[
        filtered[service_column].astype(str).str.casefold() == query.service.casefold()
    ]
    incident_rows = filtered[
        (filtered["_timestamp_seconds"] >= query.start_time.timestamp())
        & (filtered["_timestamp_seconds"] <= query.end_time.timestamp())
    ]
    if incident_rows.empty:
        return _result(query, QueryStatus.SUCCESS_EMPTY, (), started_at)

    def select_error_rows(rows: pd.DataFrame) -> pd.DataFrame:
        messages = rows[message_column].astype(str)
        if severity_column is not None:
            severities = rows[severity_column].astype(str).str.upper()
            mask = severities.isin({"WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"})
        else:
            severities = pd.Series("UNKNOWN", index=rows.index)
            mask = messages.str.contains(_ERROR_WORDS, na=False)
        selected = rows[mask].copy()
        selected["_template"] = messages[mask].map(
            lambda message: _VARIABLE_TOKEN.sub("<*>", message).strip()
        )
        selected["_severity"] = severities[mask]
        return selected

    incident_errors = select_error_rows(incident_rows)
    if incident_errors.empty:
        return _result(query, QueryStatus.SUCCESS_EMPTY, (), started_at)

    if (query.baseline_start is None) != (query.baseline_end is None):
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "both log baseline bounds are required",
        )
    baseline_errors: pd.DataFrame | None = None
    baseline_duration = 0.0
    if query.baseline_start is not None and query.baseline_end is not None:
        baseline_rows = filtered[
            (filtered["_timestamp_seconds"] >= query.baseline_start.timestamp())
            & (filtered["_timestamp_seconds"] < query.baseline_end.timestamp())
        ]
        if baseline_rows.empty:
            return _result(
                query,
                QueryStatus.PARTIAL,
                (),
                started_at,
                "log baseline window has no service-scoped samples",
            )
        baseline_errors = select_error_rows(baseline_rows)
        baseline_duration = (query.baseline_end - query.baseline_start).total_seconds()

    incident_duration = (query.end_time - query.start_time).total_seconds()
    summaries: list[LogEventSummary] = []
    for (template, severity), group in incident_errors.groupby(
        ["_template", "_severity"], sort=True, dropna=False
    ):
        baseline_count: int | None = None
        rate_increase: float | None = None
        anomalous = False
        if baseline_errors is not None:
            baseline_count = len(
                baseline_errors[
                    (baseline_errors["_template"] == template)
                    & (baseline_errors["_severity"] == severity)
                ]
            )
            rate_increase = len(group) / max(incident_duration, 1e-12) - baseline_count / max(
                baseline_duration, 1e-12
            )
            anomalous = rate_increase > 0
        timestamps = group["_timestamp_seconds"]
        summaries.append(
            LogEventSummary(
                service=query.service,
                template=str(template),
                count=len(group),
                baseline_count=baseline_count,
                incident_count=len(group),
                rate_increase=rate_increase,
                anomalous=anomalous,
                severity=str(severity),
                first_timestamp=float(timestamps.min()),
                last_timestamp=float(timestamps.max()),
            )
        )
    return _result(query, QueryStatus.SUCCESS_WITH_DATA, tuple(summaries), started_at)


def _record_observation(
    record: LogEventSummary | dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    if isinstance(record, LogEventSummary):
        observation = (
            f"{record.service} emitted template {record.template!r} {record.count} time(s) "
            f"at severity {record.severity}."
        )
        return (
            record.service,
            observation,
            {
                **record.model_dump(),
                "signal_present": True,
                "anomaly": record.anomalous,
                "anomaly_rule": "baseline_rate_increase",
            },
        )
    observation = str(record.get("observation", "log query returned an incomplete summary"))
    return str(record.get("service", "unknown")), observation, dict(record)


def normalize_log_events(
    result: ToolQueryResult[LogEventSummary] | ToolQueryResult[dict[str, str]],
) -> list[EvidenceRecord]:
    """Create provenance-bearing evidence; never upgrade incomplete results to support."""

    polarity = (
        EvidencePolarity.NEUTRAL
        if result.status is QueryStatus.SUCCESS_WITH_DATA
        else EvidencePolarity.UNKNOWN
    )
    source_reference = result.source_refs[0] if result.source_refs else "logs:unavailable"
    evidence: list[EvidenceRecord] = []
    for index, raw_record in enumerate(result.records):
        record = cast(LogEventSummary | dict[str, Any], raw_record)
        service, observation, attributes = _record_observation(record)
        evidence.append(
            EvidenceRecord(
                evidence_id=str(
                    uuid5(NAMESPACE_URL, f"{result.query_id}:logs:{index}:{observation}")
                ),
                modality=Modality.LOGS,
                service=service,
                observation=observation,
                query_id=result.query_id,
                query_status=result.status,
                polarity=polarity,
                source_reference=f"{source_reference}#query={result.query_id}",
                attributes=attributes,
                created_at=result.ended_at,
            )
        )
    return evidence
