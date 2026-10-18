"""Closed vocabularies used across the diagnostic workflow."""

from enum import StrEnum


class QueryStatus(StrEnum):
    """Outcome of one telemetry query without collapsing missingness into failure."""

    SUCCESS_WITH_DATA = "SUCCESS_WITH_DATA"
    SUCCESS_EMPTY = "SUCCESS_EMPTY"
    PARTIAL = "PARTIAL"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    TOOL_ERROR = "TOOL_ERROR"
    TIMEOUT = "TIMEOUT"


class EvidencePolarity(StrEnum):
    """Relationship between an observation and a diagnostic hypothesis."""

    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class HypothesisStatus(StrEnum):
    """Lifecycle state retained for every diagnostic hypothesis."""

    ACTIVE = "ACTIVE"
    DEFERRED = "DEFERRED"
    REFUTED = "REFUTED"
    LEADING = "LEADING"

    # Transitional aliases keep the existing graph runnable until the MAS graph replaces it.
    PROPOSED = ACTIVE
    UNDER_INVESTIGATION = ACTIVE
    SUPPORTED = LEADING
    UNRESOLVED = DEFERRED


class CausalRole(StrEnum):
    """Position of a candidate within the incident's causal chain."""

    INITIATING_CAUSE = "INITIATING_CAUSE"
    CONTRIBUTING_FAULT = "CONTRIBUTING_FAULT"
    PROPAGATED_SYMPTOM = "PROPAGATED_SYMPTOM"
    UNKNOWN = "UNKNOWN"


class TaskKind(StrEnum):
    """Supervisor-level investigation shape."""

    COMPARISON = "COMPARISON"
    CHALLENGE = "CHALLENGE"


class DiagnosisStatus(StrEnum):
    """Terminal diagnostic outcome."""

    CONFIRMED = "CONFIRMED"
    NARROWED = "NARROWED"
    INDETERMINATE = "INDETERMINATE"
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"


class Modality(StrEnum):
    """Telemetry modalities supported by the first prototype."""

    LOGS = "logs"
    METRICS = "metrics"
    TRACES = "traces"


class TaskStatus(StrEnum):
    """Lifecycle state of a bounded investigation task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
