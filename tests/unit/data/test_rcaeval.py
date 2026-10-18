import json
import shutil
from pathlib import Path

import pytest

from poise_rca.data import discover_re2_ob, load_re2_ob_case, to_agent_case_context
from poise_rca.domain import Modality

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic_case"


def _labeled_case(tmp_path: Path) -> Path:
    case_path = tmp_path / "re2ob_checkoutservice_delay_1"
    shutil.copytree(FIXTURE, case_path)
    return case_path


def test_agent_case_context_excludes_ground_truth(tmp_path: Path) -> None:
    _labeled_case(tmp_path)
    record = discover_re2_ob(tmp_path)[0]
    bundle = load_re2_ob_case(record)

    serialized = to_agent_case_context(bundle).model_dump_json().lower()

    assert "checkoutservice" not in serialized
    assert "delay" not in serialized
    assert str(record.source_locator).lower() not in serialized


def test_incident_window_is_timezone_aware(tmp_path: Path) -> None:
    _labeled_case(tmp_path)
    bundle = load_re2_ob_case(discover_re2_ob(tmp_path)[0])
    context = bundle.agent_view()

    assert context.incident_start.utcoffset() is not None
    assert context.incident_end.utcoffset() is not None
    assert context.incident_start < context.incident_end
    assert context.available_modalities == frozenset(
        {Modality.LOGS, Modality.METRICS, Modality.TRACES}
    )


def test_safe_case_id_is_stable_across_dataset_locations(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    shutil.copytree(FIXTURE, first / "re2ob_checkoutservice_delay_1")
    shutil.copytree(FIXTURE, second / "re2ob_checkoutservice_delay_1")

    first_record = discover_re2_ob(first)[0]
    second_record = discover_re2_ob(second)[0]

    assert first_record.safe_case_id == second_record.safe_case_id
    assert "checkoutservice" not in first_record.safe_case_id
    assert "delay" not in first_record.safe_case_id


def test_camel_case_rcaeval_columns_are_normalized(tmp_path: Path) -> None:
    case_path = tmp_path / "re2ob_checkoutservice_delay_1"
    case_path.mkdir()
    (case_path / "inject_time.txt").write_text("1735689600", encoding="utf-8")
    (case_path / "traces.csv").write_text(
        "timestamp,traceID,spanID,parentSpanID,serviceName,duration_ms,statusCode\n"
        "1735689600,t1,s1,,frontend,10,OK\n",
        encoding="utf-8",
    )

    bundle = load_re2_ob_case(discover_re2_ob(tmp_path)[0])

    assert bundle.traces is not None
    assert {"trace_id", "span_id", "parent_span_id", "service_name", "status_code"} <= set(
        bundle.traces.columns
    )


def test_loader_rejects_manifest_checksum_drift(tmp_path: Path) -> None:
    case_path = _labeled_case(tmp_path)
    record = discover_re2_ob(tmp_path)[0]
    (case_path / "logs.csv").write_text("timestamp,message\n1,changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_re2_ob_case(record)


def test_loader_rejects_unlisted_telemetry_file(tmp_path: Path) -> None:
    _labeled_case(tmp_path)
    record = discover_re2_ob(tmp_path)[0]
    incomplete = record.model_copy(
        update={"checksums": {"inject_time.txt": record.checksums["inject_time.txt"]}}
    )

    with pytest.raises(ValueError, match="manifest checksum coverage mismatch"):
        load_re2_ob_case(incomplete)


def test_original_metrics_json_is_loaded_as_wide_time_series(tmp_path: Path) -> None:
    case_path = tmp_path / "re2ob_checkoutservice_delay_1"
    case_path.mkdir()
    (case_path / "inject_time.txt").write_text("1735689600", encoding="utf-8")
    (case_path / "metrics.json").write_text(
        json.dumps(
            {
                "checkoutservice_cpu": [[1735689540, 0.2], [1735689600, 0.9]],
                "frontend_latency": [[1735689540, 120], [1735689660, 780]],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_re2_ob_case(discover_re2_ob(tmp_path)[0])

    assert bundle.metrics is not None
    assert list(bundle.metrics.columns) == [
        "timestamp",
        "checkoutservice_cpu",
        "frontend_latency",
    ]
    assert bundle.metrics["timestamp"].tolist() == [1735689540, 1735689600, 1735689660]
    assert bundle.metrics["checkoutservice_cpu"].isna().sum() == 1


def test_original_nested_layout_and_flat_layout_have_same_safe_id(tmp_path: Path) -> None:
    flat_root = tmp_path / "flat"
    flat_root.mkdir()
    shutil.copytree(FIXTURE, flat_root / "re2ob_checkoutservice_delay_1")
    nested_root = tmp_path / "nested" / "RE2" / "RE2-OB"
    nested_case = nested_root / "checkoutservice_delay" / "1"
    nested_case.parent.mkdir(parents=True)
    shutil.copytree(FIXTURE, nested_case)

    flat = discover_re2_ob(flat_root)
    nested = discover_re2_ob(nested_root)

    assert len(flat) == len(nested) == 1
    assert flat[0].safe_case_id == nested[0].safe_case_id
    assert nested[0].root_cause_service == "checkoutservice"
    assert nested[0].fault_type == "delay"
    assert nested[0].instance == 1
    assert nested[0].source_locator == nested_case.resolve()
