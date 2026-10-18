"""RCAEval RE2-OB discovery and label-safe telemetry loading."""

import json
import re
from collections.abc import Hashable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd

from poise_rca.data.models import CaseRecord, TelemetryBundle
from poise_rca.domain import CaseContext

_FLAT_CASE_NAME = re.compile(
    r"^re2[-_]?ob_(?P<service>.+)_(?P<fault>cpu|mem|disk|delay|loss|socket)_"
    r"(?P<instance>\d+)$",
    re.IGNORECASE,
)
_NESTED_CASE_GROUP = re.compile(
    r"^(?P<service>.+)_(?P<fault>cpu|mem|disk|delay|loss|socket)$",
    re.IGNORECASE,
)
_TELEMETRY_FILES = (
    "metrics.parquet",
    "metrics.json",
    "metrics.csv",
    "logs.parquet",
    "logs.csv",
    "traces.parquet",
    "traces.csv",
    "inject_time.txt",
)
_TIME_ALIASES = {"time", "timestamp", "ts", "time_unix", "time_unix_nano"}
_COMPACT_IDENTIFIER_COLUMNS = {"traceid", "spanid", "parentspanid"}
_COLUMN_ALIASES = {
    "containername": "container_name",
    "parentspanid": "parent_span_id",
    "servicename": "service_name",
    "spanid": "span_id",
    "statuscode": "status_code",
    "traceid": "trace_id",
}


def _safe_case_id(case_name: str) -> str:
    digest = sha256(f"RCAEval:RE2-OB:{case_name.lower()}".encode()).hexdigest()[:16]
    return f"re2-ob-{digest}"


def _case_descriptor(case_path: Path) -> tuple[str, str, int, str] | None:
    flat = _FLAT_CASE_NAME.match(case_path.name)
    if flat is not None:
        service = flat.group("service").lower()
        fault = flat.group("fault").lower()
        instance = int(flat.group("instance"))
        return service, fault, instance, f"re2ob_{service}_{fault}_{instance}"

    if not case_path.name.isdigit():
        return None
    group = _NESTED_CASE_GROUP.match(case_path.parent.name)
    has_re2_ob_ancestor = any(
        re.sub(r"[^a-z0-9]+", "", part.casefold()) == "re2ob" for part in case_path.parts
    )
    if group is None or not has_re2_ob_ancestor:
        return None
    service = group.group("service").lower()
    fault = group.group("fault").lower()
    instance = int(case_path.name)
    return service, fault, instance, f"re2ob_{service}_{fault}_{instance}"


def _file_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_re2_ob(root: Path) -> list[CaseRecord]:
    """Discover original or Parquet RE2-OB case directories deterministically."""

    root = root.resolve()
    candidates = [root] if root.is_dir() else []
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    records: list[CaseRecord] = []
    for case_path in candidates:
        descriptor = _case_descriptor(case_path)
        if descriptor is None or not (case_path / "inject_time.txt").is_file():
            continue
        service, fault, instance, canonical_name = descriptor
        checksums = {
            name: _file_checksum(case_path / name)
            for name in _TELEMETRY_FILES
            if (case_path / name).is_file()
        }
        records.append(
            CaseRecord(
                safe_case_id=_safe_case_id(canonical_name),
                source_locator=case_path.resolve(),
                dataset_version="RCAEval-RE2-OB",
                root_cause_service=service,
                fault_type=fault,
                instance=instance,
                checksums=checksums,
            )
        )
    return sorted(records, key=lambda record: record.safe_case_id)


def _read_csv_preserving_ids(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    identifier_types: dict[Hashable, str] = {
        column: "string"
        for column in columns
        if re.sub(r"[^a-z0-9]+", "", str(column).strip().lower()) in _COMPACT_IDENTIFIER_COLUMNS
    }
    return pd.read_csv(path, dtype=identifier_types or None)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in normalized.columns
    ]
    normalized = normalized.rename(
        columns={column: _COLUMN_ALIASES.get(column, column) for column in normalized.columns}
    )
    if len(set(normalized.columns)) != len(normalized.columns):
        raise ValueError("telemetry columns collide after normalization")
    timestamp_columns = [column for column in normalized.columns if column in _TIME_ALIASES]
    if timestamp_columns:
        normalized = normalized.rename(columns={timestamp_columns[0]: "timestamp"})
    return normalized


def _read_metrics_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("metrics JSON must map metric names to timestamp/value pairs")
    metric_values: dict[str, dict[int | float, object]] = {}
    timestamps: set[int | float] = set()
    for metric, raw_points in raw.items():
        if not isinstance(metric, str) or not isinstance(raw_points, Sequence):
            raise ValueError("metrics JSON contains an invalid metric series")
        points: dict[int | float, object] = {}
        for raw_point in raw_points:
            if (
                not isinstance(raw_point, Sequence)
                or isinstance(raw_point, (str, bytes))
                or len(raw_point) != 2
                or not isinstance(raw_point[0], (int, float))
            ):
                raise ValueError("metrics JSON points must be [timestamp, value] pairs")
            timestamp = raw_point[0]
            points[timestamp] = raw_point[1]
            timestamps.add(timestamp)
        if points:
            metric_values[metric] = points
    ordered_timestamps = sorted(timestamps)
    frame = pd.DataFrame({"time": ordered_timestamps})
    for metric, points in metric_values.items():
        frame[metric] = frame["time"].map(points)
    return frame


def _read_table(case_path: Path, stem: str) -> pd.DataFrame | None:
    for suffix in ("parquet", "json", "csv"):
        path = case_path / f"{stem}.{suffix}"
        if not path.is_file():
            continue
        if suffix == "parquet":
            frame = pd.read_parquet(path)
        elif suffix == "json":
            frame = _read_metrics_json(path) if stem == "metrics" else pd.read_json(path)
        else:
            frame = _read_csv_preserving_ids(path)
        return _normalize_columns(frame)
    return None


def _read_injection_time(path: Path) -> datetime:
    raw_timestamp = float(path.read_text(encoding="utf-8").strip())
    while raw_timestamp > 40_000_000_000:
        raw_timestamp /= 1000
    return datetime.fromtimestamp(raw_timestamp, tz=UTC)


def load_re2_ob_case(record: CaseRecord) -> TelemetryBundle:
    """Load telemetry while keeping labels and raw paths outside the returned bundle."""

    actual_files = {
        filename for filename in _TELEMETRY_FILES if (record.source_locator / filename).is_file()
    }
    if set(record.checksums) != actual_files:
        raise ValueError(f"manifest checksum coverage mismatch for case {record.safe_case_id}")
    for filename, expected in record.checksums.items():
        path = record.source_locator / filename
        if not path.is_file() or _file_checksum(path) != expected:
            raise ValueError(f"checksum mismatch for {filename!r} in case {record.safe_case_id}")
    injection_time = _read_injection_time(record.source_locator / "inject_time.txt")
    return TelemetryBundle(
        safe_case_id=record.safe_case_id,
        incident_start=injection_time - timedelta(minutes=5),
        incident_end=injection_time + timedelta(minutes=5),
        initial_symptom="Service health degradation detected near the incident window.",
        entry_service=None,
        metrics=_read_table(record.source_locator, "metrics"),
        logs=_read_table(record.source_locator, "logs"),
        traces=_read_table(record.source_locator, "traces"),
    )


def to_agent_case_context(bundle: TelemetryBundle) -> CaseContext:
    """Convert loaded telemetry metadata to the label-safe graph input."""

    return bundle.agent_view()
