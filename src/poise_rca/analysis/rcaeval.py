"""RCAEval Multi-source BARO adapter for label-safe telemetry bundles."""

import re
from collections.abc import Callable, Mapping
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from poise_rca.data import TelemetryBundle
from poise_rca.domain import (
    AnomalyFinding,
    CaseContext,
    Modality,
    QueryStatus,
    StaticAnalysisResult,
)

MMBaroBackend = Callable[..., Mapping[str, object]]
_TIME_COLUMNS = ("time", "timestamp", "ts")
_SERVICE_COLUMNS = ("service", "service_name", "container_name")
_LOG_MESSAGE_COLUMNS = ("template", "message", "body")
_TRACE_TARGET_COLUMNS = ("peer_service", "target_service")
_TRACE_STATUS_COLUMNS = ("status", "status_code")


def _default_backend(
    data: dict[str, pd.DataFrame], inject_time: float, *, dataset: str
) -> Mapping[str, object]:
    from RCAEval.e2e import mmbaro  # type: ignore[import-untyped]

    result = mmbaro(data, inject_time, dataset=dataset)
    if not isinstance(result, Mapping):
        raise TypeError("RCAEval mmbaro must return a mapping")
    return result


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.astype("int64") / 1_000_000_000
    return pd.to_numeric(values, errors="coerce")


def _time_column(frame: pd.DataFrame) -> str | None:
    return _first_column(frame, _TIME_COLUMNS)


def _metric_input(
    frame: pd.DataFrame | None,
    inventory: frozenset[str],
) -> tuple[pd.DataFrame, dict[str, tuple[str, Modality] | None]]:
    if frame is None:
        return pd.DataFrame(columns=["time"]), {}
    time_column = _time_column(frame)
    if time_column is None:
        return pd.DataFrame(columns=["time"]), {}
    prepared = frame.rename(columns={time_column: "time"}).copy()
    prepared["time"] = _seconds(prepared["time"])
    prepared = prepared.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    mapping: dict[str, tuple[str, Modality] | None] = {}
    for column in prepared.columns:
        if column == "time":
            continue
        name = str(column)
        service = next(
            (
                candidate
                for candidate in sorted(inventory, key=len, reverse=True)
                if name == candidate or name.startswith(f"{candidate}_")
            ),
            None,
        )
        mapping[name] = (service, Modality.METRICS) if service is not None else None
    return prepared, mapping


def _normalize_template(value: object) -> str:
    text = re.sub(r"\b\d+\b", "<*>", str(value).strip())
    return re.sub(r"\s+", " ", text) or "<empty>"


def _pivot_signal_counts(
    frame: pd.DataFrame,
    *,
    time_column: str,
    signal_column: str,
) -> pd.DataFrame:
    working = frame[[time_column, signal_column]].copy()
    working["time"] = (_seconds(working[time_column]) // 15 * 15).astype("Int64")
    working = working.dropna(subset=["time"])
    if working.empty:
        return pd.DataFrame(columns=["time"])
    counts = working.groupby(["time", signal_column], sort=True).size().unstack(fill_value=0)
    counts.columns.name = None
    return counts.reset_index().sort_values("time").reset_index(drop=True)


def _log_input(
    frame: pd.DataFrame | None,
    inventory: frozenset[str],
) -> tuple[pd.DataFrame, dict[str, tuple[str, Modality]]]:
    if frame is None:
        return pd.DataFrame(columns=["time"]), {}
    time_column = _time_column(frame)
    service_column = _first_column(frame, _SERVICE_COLUMNS)
    message_column = _first_column(frame, _LOG_MESSAGE_COLUMNS)
    if time_column is None or service_column is None or message_column is None:
        return pd.DataFrame(columns=["time"]), {}
    working = frame.copy()
    working["_service"] = working[service_column].astype(str).str.strip()
    working = working[working["_service"].isin(inventory)]
    working["_signal"] = working.apply(
        lambda row: f"log::{row['_service']}::{_normalize_template(row[message_column])}",
        axis=1,
    )
    prepared = _pivot_signal_counts(
        working,
        time_column=time_column,
        signal_column="_signal",
    )
    mapping = {
        str(column): (str(column).split("::", 2)[1], Modality.LOGS)
        for column in prepared.columns
        if column != "time"
    }
    return prepared, mapping


def _trace_inputs(
    frame: pd.DataFrame | None,
    inventory: frozenset[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[str, Modality]],
]:
    if frame is None:
        empty = pd.DataFrame(columns=["time"])
        return empty, empty.copy(), {}
    time_column = _time_column(frame)
    service_column = _first_column(frame, _SERVICE_COLUMNS)
    target_column = _first_column(frame, _TRACE_TARGET_COLUMNS)
    if time_column is None or service_column is None:
        empty = pd.DataFrame(columns=["time"])
        return empty, empty.copy(), {}
    working = frame.copy()
    working["_service"] = working[service_column].astype(str).str.strip()
    working = working[working["_service"].isin(inventory)]
    if target_column is None:
        working["_target"] = "unknown"
    else:
        working["_target"] = working[target_column].astype(str).str.strip()
    working["time"] = (_seconds(working[time_column]) // 15 * 15).astype("Int64")
    working = working.dropna(subset=["time"])
    working["_edge"] = working["_service"] + "->" + working["_target"]

    status_column = _first_column(working, _TRACE_STATUS_COLUMNS)
    if status_column is None:
        working["_error"] = 0
    else:
        working["_error"] = (
            working[status_column]
            .astype(str)
            .str.contains(r"error|fail", case=False, regex=True)
            .astype(int)
        )
    error_pivot = working.pivot_table(
        index="time", columns="_edge", values="_error", aggfunc="sum", fill_value=0
    )
    error_pivot.columns = [f"trace_error::{column}" for column in error_pivot.columns]
    error_frame = error_pivot.reset_index().sort_values("time").reset_index(drop=True)

    if "duration_ms" in working.columns:
        latency_pivot = working.pivot_table(
            index="time", columns="_edge", values="duration_ms", aggfunc="max"
        )
        latency_pivot.columns = [f"trace_latency::{column}" for column in latency_pivot.columns]
        latency_frame = latency_pivot.reset_index().sort_values("time").reset_index(drop=True)
    else:
        latency_frame = pd.DataFrame({"time": error_frame["time"]})

    mapping: dict[str, tuple[str, Modality]] = {}
    for column in (*error_frame.columns, *latency_frame.columns):
        if column == "time":
            continue
        edge = str(column).split("::", 1)[1]
        mapping[str(column)] = (edge.split("->", 1)[0], Modality.TRACES)
    return error_frame, latency_frame, mapping


class RCAEvalMMBaroAnalyzer:
    """Prepare multi-source time series and normalize RCAEval's ranked indicators."""

    def __init__(self, backend: MMBaroBackend | None = None) -> None:
        self._backend = backend or _default_backend

    def analyze(
        self,
        case: CaseContext,
        telemetry: TelemetryBundle,
        service_inventory: frozenset[str],
    ) -> StaticAnalysisResult:
        metric, metric_mapping = _metric_input(telemetry.metrics, service_inventory)
        logts, log_mapping = _log_input(telemetry.logs, service_inventory)
        tracets_err, tracets_lat, trace_mapping = _trace_inputs(telemetry.traces, service_inventory)
        prepared = {
            "metric": metric,
            "logts": logts,
            "tracets_err": tracets_err,
            "tracets_lat": tracets_lat,
        }
        statuses = {
            Modality.METRICS: self._status(telemetry.metrics, metric),
            Modality.LOGS: self._status(telemetry.logs, logts),
            Modality.TRACES: self._trace_status(telemetry.traces, tracets_err, tracets_lat),
        }
        injection_time = case.incident_start + (case.incident_end - case.incident_start) / 2
        try:
            raw_result = self._backend(
                prepared,
                injection_time.timestamp(),
                dataset="re2-ob",
            )
        except Exception as exc:
            errors = {
                modality: f"RCAEval mmbaro failed: {type(exc).__name__}: {exc}"
                for modality, status in statuses.items()
                if status is not QueryStatus.DATA_UNAVAILABLE
            }
            failed_statuses = {
                modality: (
                    QueryStatus.TOOL_ERROR if status is not QueryStatus.DATA_UNAVAILABLE else status
                )
                for modality, status in statuses.items()
            }
            return StaticAnalysisResult(statuses=failed_statuses, errors=errors)

        ranks = raw_result.get("ranks", ())
        if not isinstance(ranks, (list, tuple)):
            raise TypeError("RCAEval mmbaro result 'ranks' must be a sequence")
        raw_scores = raw_result.get("scores", {})
        scores = raw_scores if isinstance(raw_scores, Mapping) else {}
        indicator_mapping: dict[str, tuple[str, Modality] | None] = {
            **metric_mapping,
            **log_mapping,
            **trace_mapping,
        }
        findings: list[AnomalyFinding] = []
        unmapped: list[str] = []
        for rank, raw_indicator in enumerate(ranks, start=1):
            indicator = str(raw_indicator)
            target = indicator_mapping.get(indicator)
            if target is None:
                unmapped.append(indicator)
                continue
            service, modality = target
            raw_score = scores.get(indicator)
            score = float(raw_score) if isinstance(raw_score, (int, float)) else None
            findings.append(
                AnomalyFinding(
                    finding_id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"poise-mmbaro:{case.case_id}:{rank}:{indicator}",
                        )
                    ),
                    service=service,
                    modality=modality,
                    signal=indicator,
                    observation=f"Multi-source BARO ranked {indicator} at position {rank}.",
                    source_reference=(
                        f"case:{case.case_id}:static:mmbaro:{modality.value}:{indicator}"
                    ),
                    analyzer="RCAEval.Multi-source-BARO",
                    score=score,
                    rank=rank,
                )
            )
        return StaticAnalysisResult(
            findings=tuple(findings),
            statuses=statuses,
            unmapped_indicators=tuple(unmapped),
        )

    @staticmethod
    def _status(raw: pd.DataFrame | None, prepared: pd.DataFrame) -> QueryStatus:
        if raw is None:
            return QueryStatus.DATA_UNAVAILABLE
        return (
            QueryStatus.SUCCESS_WITH_DATA
            if len(prepared.columns) > 1 and not prepared.empty
            else QueryStatus.SUCCESS_EMPTY
        )

    @staticmethod
    def _trace_status(
        raw: pd.DataFrame | None,
        errors: pd.DataFrame,
        latency: pd.DataFrame,
    ) -> QueryStatus:
        if raw is None:
            return QueryStatus.DATA_UNAVAILABLE
        has_data = any(len(frame.columns) > 1 and not frame.empty for frame in (errors, latency))
        return QueryStatus.SUCCESS_WITH_DATA if has_data else QueryStatus.SUCCESS_EMPTY
