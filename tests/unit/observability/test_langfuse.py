from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from poise_rca.observability.langfuse import (
    LangFuseSettings,
    NoOpTraceRuntime,
    create_trace_runtime,
)


class FakeClient:
    def __init__(self, *, fail_flush: bool = False, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.fail_flush = fail_flush
        self.observations: list[dict[str, object]] = []
        self.outputs: list[object] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs: object) -> Iterator[object]:
        self.observations.append(kwargs)
        yield object()

    def update_current_span(self, *, output: object) -> None:
        self.outputs.append(output)

    def flush(self) -> None:
        if self.fail_flush:
            raise RuntimeError("collector unavailable")

    def get_current_trace_id(self) -> str | None:
        return "trace-test"


def _settings() -> LangFuseSettings:
    return LangFuseSettings(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="https://langfuse.invalid",
        environment="test",
    )


def test_no_credentials_returns_noop_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_constructor(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("client must not be constructed")

    monkeypatch.setattr("poise_rca.observability.langfuse.Langfuse", fail_constructor)

    runtime = create_trace_runtime(LangFuseSettings())

    assert isinstance(runtime, NoOpTraceRuntime)
    assert runtime.callbacks == ()
    with runtime.observation("noop", "chain", metadata={"case_id": "safe"}):
        pass
    runtime.set_trace_output({"status": "INDETERMINATE"})
    runtime.flush()


def test_client_construction_failure_degrades_to_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_constructor(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("bad endpoint")

    monkeypatch.setattr("poise_rca.observability.langfuse.Langfuse", fail_constructor)

    with pytest.warns(RuntimeWarning, match="LangFuse initialization failed"):
        runtime = create_trace_runtime(_settings())

    assert isinstance(runtime, NoOpTraceRuntime)


def test_runtime_uses_explicit_observation_types_and_omits_raw_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr(
        "poise_rca.observability.langfuse.Langfuse",
        lambda **kwargs: client,
    )
    monkeypatch.setattr(
        "poise_rca.observability.langfuse.CallbackHandler",
        lambda **kwargs: ("callback", kwargs),
    )
    runtime = create_trace_runtime(_settings())
    metadata: dict[str, Any] = {
        "case_id": "re2-ob-safe",
        "evidence_ids": ["e-1"],
        "raw_logs": [{"message": "secret raw log"}],
        "records": [{"timestamp": 1, "value": 99}],
    }

    for observation_type in ("tool", "agent", "chain", "evaluator"):
        with runtime.observation(
            f"{observation_type}.step",
            observation_type,
            metadata=metadata,
        ):
            pass

    assert [item["as_type"] for item in client.observations] == [
        "tool",
        "agent",
        "chain",
        "evaluator",
    ]
    serialized = repr(client.observations)
    assert "re2-ob-safe" in serialized
    assert "secret raw log" not in serialized
    assert "timestamp" not in serialized


def test_export_failures_warn_and_do_not_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(fail_flush=True)
    monkeypatch.setattr(
        "poise_rca.observability.langfuse.Langfuse",
        lambda **kwargs: client,
    )
    monkeypatch.setattr(
        "poise_rca.observability.langfuse.CallbackHandler",
        lambda **kwargs: object(),
    )
    runtime = create_trace_runtime(_settings())

    runtime.set_trace_output(
        {
            "status": "INDETERMINATE",
            "stop_reason": "insufficient evidence",
            "raw_traces": [{"trace_id": "must-not-export"}],
        }
    )
    with pytest.warns(RuntimeWarning, match="LangFuse flush failed"):
        runtime.flush()

    assert "must-not-export" not in repr(client.outputs)
