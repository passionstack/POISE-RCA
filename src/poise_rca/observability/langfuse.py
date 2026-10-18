"""Fail-open LangFuse facade with a raw-telemetry denylist."""

import sys
import warnings
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Literal, Protocol, cast

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, ConfigDict, SecretStr

ObservationType = Literal["tool", "agent", "chain", "evaluator"]
_RAW_KEYS = {
    "dataframe",
    "logs",
    "metrics",
    "raw_logs",
    "raw_metrics",
    "raw_telemetry",
    "raw_traces",
    "records",
    "telemetry",
    "traces",
}


class LangFuseSettings(BaseModel):
    """Optional tracing credentials and privacy controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    base_url: str | None = None
    environment: str = "development"
    export_raw_telemetry: bool = False


class TraceRuntime(Protocol):
    """Small tracing boundary consumed by the graph and CLI."""

    callbacks: tuple[object, ...]

    @property
    def trace_id(self) -> str | None: ...

    def observation(
        self,
        name: str,
        observation_type: ObservationType,
        *,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[object | None]: ...

    def event(
        self,
        name: str,
        attributes: dict[str, object],
        observation_type: ObservationType = "chain",
    ) -> None: ...

    def set_trace_output(self, output: object) -> None: ...

    def flush(self) -> None: ...


def _safe_payload(value: object, *, export_raw_telemetry: bool = False) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_payload(item, export_raw_telemetry=export_raw_telemetry)
            for key, item in value.items()
            if export_raw_telemetry or str(key).casefold() not in _RAW_KEYS
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_payload(item, export_raw_telemetry=export_raw_telemetry) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _safe_payload(model_dump(mode="json"), export_raw_telemetry=export_raw_telemetry)
    return repr(value)


class NoOpTraceRuntime:
    """Tracing implementation used when tracing is disabled or unavailable."""

    callbacks: tuple[object, ...] = ()

    @property
    def trace_id(self) -> str | None:
        return None

    @contextmanager
    def observation(
        self,
        name: str,
        observation_type: ObservationType,
        *,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[object | None]:
        del name, observation_type, input, metadata
        yield None

    def event(
        self,
        name: str,
        attributes: dict[str, object],
        observation_type: ObservationType = "chain",
    ) -> None:
        del name, attributes, observation_type

    def set_trace_output(self, output: object) -> None:
        del output

    def flush(self) -> None:
        return None


class _LangFuseClient(Protocol):
    def start_as_current_observation(self, **kwargs: object) -> AbstractContextManager[object]: ...

    def update_current_span(self, *, output: object) -> None: ...

    def flush(self) -> None: ...

    def get_current_trace_id(self) -> str | None: ...


class LangFuseTraceRuntime:
    """Privacy-filtered adapter over one LangFuse client."""

    def __init__(
        self,
        client: _LangFuseClient,
        callback: object,
        *,
        export_raw_telemetry: bool,
    ) -> None:
        self._client = client
        self._export_raw_telemetry = export_raw_telemetry
        self.callbacks: tuple[object, ...] = (callback,)

    @property
    def trace_id(self) -> str | None:
        try:
            return self._client.get_current_trace_id()
        except Exception:
            return None

    @contextmanager
    def observation(
        self,
        name: str,
        observation_type: ObservationType,
        *,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[object | None]:
        try:
            manager = self._client.start_as_current_observation(
                name=name,
                as_type=observation_type,
                input=_safe_payload(input, export_raw_telemetry=self._export_raw_telemetry),
                metadata=_safe_payload(
                    metadata or {},
                    export_raw_telemetry=self._export_raw_telemetry,
                ),
            )
            observation = manager.__enter__()
        except Exception as exc:
            warnings.warn(f"LangFuse observation start failed: {exc}", RuntimeWarning, stacklevel=2)
            yield None
            return

        try:
            yield observation
        except BaseException:
            manager.__exit__(*sys.exc_info())
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception as exc:
                warnings.warn(
                    f"LangFuse observation export failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def event(
        self,
        name: str,
        attributes: dict[str, object],
        observation_type: ObservationType = "chain",
    ) -> None:
        with self.observation(name, observation_type, metadata=attributes):
            pass

    def set_trace_output(self, output: object) -> None:
        try:
            self._client.update_current_span(
                output=_safe_payload(output, export_raw_telemetry=self._export_raw_telemetry)
            )
        except Exception as exc:
            warnings.warn(f"LangFuse trace output failed: {exc}", RuntimeWarning, stacklevel=2)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            warnings.warn(f"LangFuse flush failed: {exc}", RuntimeWarning, stacklevel=2)


def create_trace_runtime(settings: LangFuseSettings) -> TraceRuntime:
    """Construct tracing when fully configured, otherwise return a safe no-op."""

    if not settings.enabled or settings.public_key is None or settings.secret_key is None:
        return NoOpTraceRuntime()
    public_key = settings.public_key.get_secret_value()
    secret_key = settings.secret_key.get_secret_value()
    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=settings.base_url,
            environment=settings.environment,
        )
        callback = CallbackHandler(public_key=public_key)
    except Exception as exc:
        warnings.warn(f"LangFuse initialization failed: {exc}", RuntimeWarning, stacklevel=2)
        return NoOpTraceRuntime()
    return LangFuseTraceRuntime(
        cast(_LangFuseClient, client),
        callback,
        export_raw_telemetry=settings.export_raw_telemetry,
    )
