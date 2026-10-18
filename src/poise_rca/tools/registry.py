"""Timeout- and exception-safe registry boundary for telemetry adapters."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from queue import Empty, Queue
from threading import Thread
from typing import Any
from uuid import uuid4

from poise_rca.data import TelemetryBundle
from poise_rca.domain import (
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
)
from poise_rca.tools.logs import LogQuery, normalize_log_events, query_logs
from poise_rca.tools.metrics import MetricQuery, normalize_metric_events, query_metrics
from poise_rca.tools.traces import TraceQuery, normalize_trace_events, query_traces

ToolHandler = Callable[[InvestigationTask], object]


class ToolRegistry:
    """Execute one bounded task and convert all boundary failures to explicit status."""

    def __init__(
        self,
        bundle: TelemetryBundle | None,
        timeout_seconds: float = 30.0,
        source_references: Mapping[Modality, str] | None = None,
        handlers: Mapping[str, ToolHandler] | None = None,
    ) -> None:
        self._bundle = bundle
        self._timeout_seconds = timeout_seconds
        self._source_references = dict(source_references or {})
        self._handlers = (
            dict(handlers)
            if handlers is not None
            else {
                "query_logs": self._run_logs,
                "query_metrics": self._run_metrics,
                "query_traces": self._run_traces,
            }
        )

    def _source(self, modality: Modality) -> str:
        return self._source_references.get(modality, f"{modality.value}:in-memory")

    def _require_bundle(self) -> TelemetryBundle:
        if self._bundle is None:
            raise RuntimeError("a telemetry bundle is required for built-in adapters")
        return self._bundle

    @staticmethod
    def _query_scope(task: InvestigationTask) -> tuple[str, datetime, datetime]:
        if task.service is None:
            raise ValueError("a built-in telemetry query requires a service")
        if task.start_time is None or task.end_time is None:
            raise ValueError("a built-in telemetry query requires start_time and end_time")
        return task.service, task.start_time, task.end_time

    def _run_logs(self, task: InvestigationTask) -> InvestigationResult:
        service, start_time, end_time = self._query_scope(task)
        duration = end_time - start_time
        query = LogQuery(
            query_id=str(uuid4()),
            service=service,
            start_time=start_time,
            end_time=end_time,
            baseline_start=start_time - duration,
            baseline_end=start_time,
            source_reference=self._source(Modality.LOGS),
        )
        result = query_logs(self._require_bundle(), query)
        return InvestigationResult(
            task_id=task.task_id,
            query_id=query.query_id,
            status=result.status,
            evidence=tuple(normalize_log_events(result)),
            error=result.error,
            started_at=result.started_at,
            finished_at=result.ended_at,
        )

    def _run_metrics(self, task: InvestigationTask) -> InvestigationResult:
        service, start_time, end_time = self._query_scope(task)
        duration = end_time - start_time
        arguments: dict[str, Any] = dict(task.arguments)
        query = MetricQuery(
            query_id=str(uuid4()),
            service=service,
            start_time=start_time,
            end_time=end_time,
            baseline_start=arguments.get("baseline_start", start_time - duration),
            baseline_end=arguments.get("baseline_end", start_time),
            metric_name=arguments.get("metric_name"),
            min_relative_change=arguments.get("min_relative_change", 0.2),
            source_reference=self._source(Modality.METRICS),
        )
        result = query_metrics(self._require_bundle(), query)
        return InvestigationResult(
            task_id=task.task_id,
            query_id=query.query_id,
            status=result.status,
            evidence=tuple(normalize_metric_events(result)),
            error=result.error,
            started_at=result.started_at,
            finished_at=result.ended_at,
        )

    def _run_traces(self, task: InvestigationTask) -> InvestigationResult:
        service, start_time, end_time = self._query_scope(task)
        duration = end_time - start_time
        query = TraceQuery(
            query_id=str(uuid4()),
            service=service,
            start_time=start_time,
            end_time=end_time,
            baseline_start=start_time - duration,
            baseline_end=start_time,
            source_reference=self._source(Modality.TRACES),
        )
        result = query_traces(self._require_bundle(), query)
        return InvestigationResult(
            task_id=task.task_id,
            query_id=query.query_id,
            status=result.status,
            evidence=tuple(normalize_trace_events(result)),
            error=result.error,
            started_at=result.started_at,
            finished_at=result.ended_at,
        )

    def execute(self, task: InvestigationTask) -> InvestigationResult:
        """Run one adapter with hard timeout and fail-closed evidence semantics."""

        started_at = datetime.now(UTC)
        query_id = str(uuid4())
        if task.tool_name is None:
            return InvestigationResult(
                task_id=task.task_id,
                query_id=query_id,
                status=QueryStatus.TOOL_ERROR,
                error="an investigation query requires a tool name",
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
        handler = self._handlers.get(task.tool_name)
        if handler is None:
            return InvestigationResult(
                task_id=task.task_id,
                query_id=query_id,
                status=QueryStatus.TOOL_ERROR,
                error=f"unknown telemetry tool: {task.tool_name}",
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )

        responses: Queue[tuple[bool, object]] = Queue(maxsize=1)

        def run_handler() -> None:
            try:
                responses.put((True, handler(task)))
            except Exception as exc:
                responses.put((False, exc))

        worker = Thread(
            target=run_handler,
            name=f"poise-tool-{task.tool_name}",
            daemon=True,
        )
        worker.start()
        try:
            succeeded, payload = responses.get(timeout=self._timeout_seconds)
        except Empty:
            return InvestigationResult(
                task_id=task.task_id,
                query_id=query_id,
                status=QueryStatus.TIMEOUT,
                error=f"tool exceeded {self._timeout_seconds:g}s timeout",
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
        if not succeeded:
            assert isinstance(payload, Exception)
            return InvestigationResult(
                task_id=task.task_id,
                query_id=query_id,
                status=QueryStatus.TOOL_ERROR,
                error=str(payload),
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
        if not isinstance(payload, InvestigationResult):
            return InvestigationResult(
                task_id=task.task_id,
                query_id=query_id,
                status=QueryStatus.TOOL_ERROR,
                error="tool handler must return InvestigationResult",
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
        return payload
