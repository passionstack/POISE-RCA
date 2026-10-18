"""Shared query contracts and telemetry-frame helpers."""

from datetime import datetime
from typing import Annotated, Self, TypeVar
from uuid import UUID

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from poise_rca.domain import QueryStatus

RecordT = TypeVar("RecordT")
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_uuid(value: str) -> str:
    UUID(value)
    return value


def _validate_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("query timestamps must be timezone-aware")
    return value


class BaseTelemetryQuery(BaseModel):
    """Fields common to every bounded local telemetry query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    service: NonEmptyStr
    start_time: datetime
    end_time: datetime
    source_reference: NonEmptyStr

    _query_id_is_uuid = field_validator("query_id")(_validate_uuid)
    _timestamps_are_aware = field_validator("start_time", "end_time")(_validate_aware)

    @model_validator(mode="after")
    def query_window_is_ordered(self) -> Self:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        return self


class ToolQueryResult[RecordT](BaseModel):
    """Lossless six-state result returned by every telemetry adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    status: QueryStatus
    records: tuple[RecordT, ...] = ()
    source_refs: tuple[str, ...] = ()
    error: str | None = None
    started_at: datetime
    ended_at: datetime
    latency_ms: float = Field(ge=0)

    _query_id_is_uuid = field_validator("query_id")(_validate_uuid)
    _timestamps_are_aware = field_validator("started_at", "ended_at")(_validate_aware)

    @model_validator(mode="after")
    def preserve_outcome_semantics(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.status is QueryStatus.SUCCESS_WITH_DATA and not self.records:
            raise ValueError("SUCCESS_WITH_DATA requires at least one record")
        if self.status is QueryStatus.SUCCESS_EMPTY and self.records:
            raise ValueError("SUCCESS_EMPTY cannot carry records")
        if (
            self.status
            in {
                QueryStatus.DATA_UNAVAILABLE,
                QueryStatus.TOOL_ERROR,
                QueryStatus.TIMEOUT,
            }
            and self.records
        ):
            raise ValueError(f"{self.status.value} cannot carry records")
        return self


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first normalized column name present in a frame."""

    return next((candidate for candidate in candidates if candidate in frame.columns), None)


def timestamp_seconds(series: pd.Series) -> pd.Series:
    """Normalize numeric or datetime-like timestamps to Unix seconds."""

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, utc=True)
        return parsed.astype("int64") / 1_000_000_000

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        median = float(numeric.dropna().abs().median())
        while median > 40_000_000_000:
            numeric = numeric / 1000
            median /= 1000
        return numeric

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.astype("int64") / 1_000_000_000


def elapsed_ms(started_at: datetime, ended_at: datetime) -> float:
    return max(0.0, (ended_at - started_at).total_seconds() * 1000)
