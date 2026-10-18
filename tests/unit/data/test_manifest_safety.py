from pathlib import Path

import pytest
from pydantic import ValidationError

from poise_rca.cli import _record_for_case
from poise_rca.data.manifest import DatasetManifest, ManifestCase


def test_manifest_rejects_path_like_case_id() -> None:
    with pytest.raises(ValidationError):
        ManifestCase(
            safe_case_id="../escape",
            source_locator="outside",
            root_cause_service="checkout",
            fault_type="delay",
            checksums={},
        )


def test_case_source_must_remain_under_configured_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    case = ManifestCase(
        safe_case_id="re2-ob-0000000000000001",
        source_locator=str(outside),
        root_cause_service="checkout",
        fault_type="delay",
        checksums={},
    )
    manifest = DatasetManifest(
        dataset_version="RCAEval-RE2-OB",
        seed=1,
        constraints={},
        cases=(case,),
    )

    with pytest.raises(ValueError, match="outside configured dataset root"):
        _record_for_case(manifest, case.safe_case_id, root)


def test_manifest_rejects_unsafe_checksum_filename() -> None:
    with pytest.raises(ValidationError):
        ManifestCase(
            safe_case_id="re2-ob-0000000000000001",
            source_locator="case",
            root_cause_service="checkout",
            fault_type="delay",
            checksums={"../outside.csv": "a" * 64},
        )
