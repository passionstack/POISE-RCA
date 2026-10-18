import time
from datetime import UTC, datetime, timedelta
from pathlib import PureWindowsPath
from uuid import uuid4

import pytest

from poise_rca.data import TelemetryBundle
from poise_rca.domain import EvidencePolarity, InvestigationTask, Modality, QueryStatus
from poise_rca.tools.base import ToolQueryResult
from poise_rca.tools.logs import LogQuery, normalize_log_events, query_logs
from poise_rca.tools.registry import ToolRegistry


@pytest.mark.parametrize("status", list(QueryStatus))
def test_query_outcomes_preserve_semantics(status: QueryStatus) -> None:
    now = datetime.now(UTC)
    result = ToolQueryResult[dict[str, str]](
        query_id=str(uuid4()),
        status=status,
        records=({"observation": "bounded clue"},)
        if status in {QueryStatus.SUCCESS_WITH_DATA, QueryStatus.PARTIAL}
        else (),
        source_refs=("telemetry.csv",),
        error="query incomplete"
        if status in {QueryStatus.PARTIAL, QueryStatus.TOOL_ERROR, QueryStatus.TIMEOUT}
        else None,
        started_at=now,
        ended_at=now + timedelta(milliseconds=2),
        latency_ms=2.0,
    )

    assert result.status is status
    assert result.model_dump()["status"] == status


def test_missing_modality_is_unavailable_on_windows_path() -> None:
    now = datetime.now(UTC)
    source = str(PureWindowsPath(r"D:\RCAEval\re2ob_cartservice_loss_1\logs.csv"))
    bundle = TelemetryBundle(
        safe_case_id="re2-ob-0000000000000001",
        incident_start=now,
        incident_end=now + timedelta(minutes=5),
        initial_symptom="Service health degradation detected.",
        entry_service=None,
        metrics=None,
        logs=None,
        traces=None,
    )
    query = LogQuery(
        query_id=str(uuid4()),
        service="frontend",
        start_time=bundle.incident_start,
        end_time=bundle.incident_end,
        source_reference=source,
    )

    result = query_logs(bundle, query)

    assert result.status is QueryStatus.DATA_UNAVAILABLE
    assert result.source_refs == (source,)
    assert normalize_log_events(result) == []


@pytest.mark.parametrize(
    "status",
    [
        QueryStatus.PARTIAL,
        QueryStatus.DATA_UNAVAILABLE,
        QueryStatus.TOOL_ERROR,
        QueryStatus.TIMEOUT,
    ],
)
def test_incomplete_query_cannot_normalize_as_support(status: QueryStatus) -> None:
    now = datetime.now(UTC)
    result = ToolQueryResult[dict[str, str]](
        query_id=str(uuid4()),
        status=status,
        records=({"observation": "untrusted clue"},) if status is QueryStatus.PARTIAL else (),
        source_refs=("logs.csv",),
        error="incomplete",
        started_at=now,
        ended_at=now,
        latency_ms=0.0,
    )

    evidence = normalize_log_events(result)

    assert all(item.polarity is not EvidencePolarity.SUPPORT for item in evidence)


def _task(tool_name: str) -> InvestigationTask:
    now = datetime.now(UTC)
    return InvestigationTask(
        task_id=str(uuid4()),
        target_hypothesis_ids=(str(uuid4()),),
        question="Does the selected signal distinguish the candidate?",
        modality=Modality.LOGS,
        service="frontend",
        start_time=now,
        end_time=now + timedelta(minutes=1),
        tool_name=tool_name,
        expected_outcomes=("an error signal is present",),
    )


def test_registry_converts_adapter_exception_to_tool_error() -> None:
    def explode(_: InvestigationTask) -> object:
        raise RuntimeError("adapter exploded")

    registry = ToolRegistry(bundle=None, handlers={"explode": explode})

    result = registry.execute(_task("explode"))

    assert result.status is QueryStatus.TOOL_ERROR
    assert result.evidence == ()
    assert "adapter exploded" in (result.error or "")


def test_registry_enforces_timeout() -> None:
    def too_slow(_: InvestigationTask) -> object:
        time.sleep(0.1)
        raise AssertionError("the registry should already have timed out")

    registry = ToolRegistry(bundle=None, timeout_seconds=0.001, handlers={"slow": too_slow})

    result = registry.execute(_task("slow"))

    assert result.status is QueryStatus.TIMEOUT
    assert result.evidence == ()


def test_registry_rejects_missing_tool_name_without_dispatch() -> None:
    task = InvestigationTask(
        task_id=str(uuid4()),
        target_hypothesis_ids=(str(uuid4()),),
        question="Which telemetry query should be used?",
    )

    result = ToolRegistry(bundle=None).execute(task)

    assert result.status is QueryStatus.TOOL_ERROR
    assert result.evidence == ()
    assert result.error == "an investigation query requires a tool name"
