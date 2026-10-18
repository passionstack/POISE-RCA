from pathlib import Path

from poise_rca.data import CaseRecord
from poise_rca.data.sampling import (
    SamplingConfig,
    SamplingConstraints,
    create_stratified_manifest,
)

FAULTS = ("cpu", "mem", "disk", "delay", "loss", "socket")
SERVICES = (
    "adservice",
    "cartservice",
    "checkoutservice",
    "emailservice",
    "recommendationservice",
)
FAULT_CATEGORY = {
    "cpu": "resource",
    "mem": "resource",
    "disk": "resource",
    "delay": "latency",
    "loss": "network",
    "socket": "network",
}


def _records(root: Path) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    number = 0
    for fault in FAULTS:
        for service in SERVICES:
            number += 1
            records.append(
                CaseRecord(
                    safe_case_id=f"re2-ob-{number:016x}",
                    source_locator=root / f"re2ob_{service}_{fault}_1",
                    dataset_version="RCAEval-RE2-OB-test",
                    root_cause_service=service,
                    fault_type=fault,
                    instance=1,
                    checksums={"metrics.csv": f"{number:064x}"},
                )
            )
    return records


def test_fixed_seed_reproduces_manifest(tmp_path: Path) -> None:
    records = _records(tmp_path)
    config = SamplingConfig(
        dataset_version="RCAEval-RE2-OB-test",
        seed=23,
        size=12,
        constraints=SamplingConstraints(samples_per_fault_type=2),
    )

    first = create_stratified_manifest(records, config)
    second = create_stratified_manifest(records, config)

    assert first == second
    safe_json = first.agent_safe_view().model_dump_json().lower()
    assert "source_locator" not in safe_json
    assert all(service not in safe_json for service in SERVICES)
    assert all(fault not in safe_json for fault in FAULTS)


def test_prototype_manifest_covers_six_faults_twice(tmp_path: Path) -> None:
    manifest = create_stratified_manifest(
        _records(tmp_path),
        SamplingConfig(
            dataset_version="RCAEval-RE2-OB-test",
            seed=7,
            size=12,
            constraints=SamplingConstraints(
                samples_per_fault_type=2,
                balance_root_cause_services=True,
            ),
        ),
    )

    counts = {fault: 0 for fault in FAULTS}
    for case in manifest.cases:
        counts[case.fault_type] += 1

    assert counts == {fault: 2 for fault in FAULTS}


def test_smoke_manifest_has_three_categories_on_distinct_services_when_available(
    tmp_path: Path,
) -> None:
    manifest = create_stratified_manifest(
        _records(tmp_path),
        SamplingConfig(
            dataset_version="RCAEval-RE2-OB-test",
            seed=11,
            size=3,
            constraints=SamplingConstraints(
                preferred_failure_categories=("resource", "latency", "network"),
                distinct_root_cause_services=True,
            ),
        ),
    )

    assert {FAULT_CATEGORY[case.fault_type] for case in manifest.cases} == {
        "resource",
        "latency",
        "network",
    }
    assert len({case.root_cause_service for case in manifest.cases}) == 3
