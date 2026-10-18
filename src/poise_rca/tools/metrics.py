"""Metric window comparison and anomaly-summary normalization."""

from datetime import UTC, datetime
from typing import Self
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from poise_rca.data import TelemetryBundle
from poise_rca.domain import EvidencePolarity, EvidenceRecord, Modality, QueryStatus
from poise_rca.tools.base import (
    BaseTelemetryQuery,
    ToolQueryResult,
    elapsed_ms,
    find_column,
    timestamp_seconds,
)


class MetricQuery(BaseTelemetryQuery):
    """Compare a pre-incident baseline with an incident metric window."""

    baseline_start: datetime
    baseline_end: datetime
    metric_name: str | None = None
    min_relative_change: float = Field(default=0.2, ge=0)

    @field_validator("baseline_start", "baseline_end")
    @classmethod
    def baseline_timestamps_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("baseline timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def baseline_window_precedes_incident(self) -> Self:
        if self.baseline_end <= self.baseline_start:
            raise ValueError("baseline_end must be later than baseline_start")
        if self.baseline_end > self.start_time:
            raise ValueError("baseline window cannot overlap the incident window")
        return self


class MetricAnomalySummary(BaseModel):
    """Baseline-versus-incident summary without raw metric samples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: str
    metric: str
    baseline_mean: float
    incident_mean: float
    absolute_delta: float
    relative_change: float
    direction: str
    baseline_samples: int
    incident_samples: int


def _result(
    query: MetricQuery,
    status: QueryStatus,
    records: tuple[MetricAnomalySummary, ...],
    started_at: datetime,
    error: str | None = None,
) -> ToolQueryResult[MetricAnomalySummary]:
    ended_at = datetime.now(UTC)
    return ToolQueryResult[MetricAnomalySummary](
        query_id=query.query_id,
        status=status,
        records=records,
        source_refs=(query.source_reference,),
        error=error,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=elapsed_ms(started_at, ended_at),
    )


def _to_long_metrics(frame: pd.DataFrame, query: MetricQuery) -> pd.DataFrame | None:
    time_column = find_column(frame, ("timestamp", "time", "ts"))
    if time_column is None:
        return None
    service_column = find_column(
        frame,
        ("service", "service_name", "servicename", "pod", "pod_name", "component"),
    )
    metric_column = find_column(frame, ("metric", "metric_name", "name"))
    value_column = find_column(frame, ("value", "metric_value"))
    if metric_column is not None and value_column is not None:
        selected = frame.copy()
        if service_column is not None:
            selected = selected[
                selected[service_column].astype(str).str.casefold() == query.service.casefold()
            ]
        selected = selected.rename(columns={metric_column: "_metric", value_column: "_value"})
        return selected[[time_column, "_metric", "_value"]]

    value_columns = [column for column in frame.columns if column != time_column]
    service_columns = [
        column for column in value_columns if query.service.casefold() in str(column).casefold()
    ]
    value_columns = service_columns
    if not value_columns:
        return pd.DataFrame(columns=[time_column, "_metric", "_value"])
    return frame.melt(
        id_vars=[time_column],
        value_vars=value_columns,
        var_name="_metric",
        value_name="_value",
    )


def query_metrics(
    bundle: TelemetryBundle,
    query: MetricQuery,
) -> ToolQueryResult[MetricAnomalySummary]:
    """Return deterministic baseline-versus-incident anomaly summaries."""

    started_at = datetime.now(UTC)
    if bundle.metrics is None:
        return _result(
            query,
            QueryStatus.DATA_UNAVAILABLE,
            (),
            started_at,
            "metric modality is unavailable",
        )
    metric_column = find_column(bundle.metrics, ("metric", "metric_name", "name"))
    value_column = find_column(bundle.metrics, ("value", "metric_value"))
    service_column = find_column(
        bundle.metrics,
        ("service", "service_name", "servicename", "pod", "pod_name", "component"),
    )
    if metric_column is not None and value_column is not None and service_column is None:
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "metric service column is missing; rows cannot be safely scoped",
        )
    metrics = _to_long_metrics(bundle.metrics, query)
    if metrics is None:
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "metric timestamp column is missing",
        )
    if metrics.empty:
        return _result(query, QueryStatus.SUCCESS_EMPTY, (), started_at)

    time_column = find_column(metrics, ("timestamp", "time", "ts"))
    if time_column is None:
        return _result(query, QueryStatus.PARTIAL, (), started_at, "timestamp is unavailable")
    metrics = metrics.assign(
        _timestamp_seconds=timestamp_seconds(metrics[time_column]),
        _numeric_value=pd.to_numeric(metrics["_value"], errors="coerce"),
    ).dropna(subset=["_timestamp_seconds", "_numeric_value"])
    if query.metric_name is not None:
        metrics = metrics[
            metrics["_metric"].astype(str).str.casefold() == query.metric_name.casefold()
        ]

    baseline = metrics[
        (metrics["_timestamp_seconds"] >= query.baseline_start.timestamp())
        & (metrics["_timestamp_seconds"] < query.baseline_end.timestamp())
    ]
    incident = metrics[
        (metrics["_timestamp_seconds"] >= query.start_time.timestamp())
        & (metrics["_timestamp_seconds"] <= query.end_time.timestamp())
    ]
    if baseline.empty:
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "metric baseline window has no usable samples",
        )
    if incident.empty:
        return _result(
            query,
            QueryStatus.PARTIAL,
            (),
            started_at,
            "metric incident window has no usable samples",
        )
    summaries: list[MetricAnomalySummary] = []
    common_metrics = sorted(
        set(baseline["_metric"].astype(str)) & set(incident["_metric"].astype(str))
    )
    for metric in common_metrics:
        baseline_values = baseline[baseline["_metric"].astype(str) == metric]["_numeric_value"]
        incident_values = incident[incident["_metric"].astype(str) == metric]["_numeric_value"]
        baseline_mean = float(baseline_values.mean())
        incident_mean = float(incident_values.mean())
        delta = incident_mean - baseline_mean
        relative_change = delta / max(abs(baseline_mean), 1e-12)
        if abs(relative_change) < query.min_relative_change:
            continue
        summaries.append(
            MetricAnomalySummary(
                service=query.service,
                metric=metric,
                baseline_mean=baseline_mean,
                incident_mean=incident_mean,
                absolute_delta=delta,
                relative_change=relative_change,
                direction="increase" if delta >= 0 else "decrease",
                baseline_samples=len(baseline_values),
                incident_samples=len(incident_values),
            )
        )
    status = QueryStatus.SUCCESS_WITH_DATA if summaries else QueryStatus.SUCCESS_EMPTY
    return _result(query, status, tuple(summaries), started_at)


def normalize_metric_events(
    result: ToolQueryResult[MetricAnomalySummary],
) -> list[EvidenceRecord]:
    """Convert metric summaries to neutral provenance-bearing evidence."""

    polarity = (
        EvidencePolarity.NEUTRAL
        if result.status is QueryStatus.SUCCESS_WITH_DATA
        else EvidencePolarity.UNKNOWN
    )
    source_reference = result.source_refs[0] if result.source_refs else "metrics:unavailable"
    evidence: list[EvidenceRecord] = []
    for index, record in enumerate(result.records):
        observation = (
            f"{record.service} metric {record.metric} changed from "
            f"{record.baseline_mean:.6g} to {record.incident_mean:.6g} "
            f"({record.direction}, relative change {record.relative_change:.3g})."
        )
        evidence.append(
            EvidenceRecord(
                evidence_id=str(
                    uuid5(NAMESPACE_URL, f"{result.query_id}:metrics:{index}:{observation}")
                ),
                modality=Modality.METRICS,
                service=record.service,
                observation=observation,
                query_id=result.query_id,
                query_status=result.status,
                polarity=polarity,
                source_reference=f"{source_reference}#query={result.query_id}",
                attributes={
                    **record.model_dump(),
                    "anomaly": True,
                    "anomaly_rule": "relative_change_threshold",
                },
                created_at=result.ended_at,
            )
        )
    return evidence
