from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from poise_rca.domain import EvidencePolarity, EvidenceRecord, Modality, QueryStatus


def _evidence(**overrides: object) -> EvidenceRecord:
    values: dict[str, object] = {
        "evidence_id": str(uuid4()),
        "modality": Modality.LOGS,
        "service": "checkout",
        "observation": "error rate increased",
        "query_id": str(uuid4()),
        "query_status": QueryStatus.SUCCESS_WITH_DATA,
        "polarity": EvidencePolarity.NEUTRAL,
        "source_reference": "logs/checkout.csv:10-20",
        "created_at": datetime.now(UTC),
    }
    values.update(overrides)
    return EvidenceRecord.model_validate(values)


def test_query_status_has_six_distinct_values() -> None:
    assert {status.value for status in QueryStatus} == {
        "SUCCESS_WITH_DATA",
        "SUCCESS_EMPTY",
        "PARTIAL",
        "DATA_UNAVAILABLE",
        "TOOL_ERROR",
        "TIMEOUT",
    }


def test_evidence_is_immutable_and_requires_source_reference() -> None:
    record = _evidence()

    with pytest.raises(ValidationError):
        record.observation = "invented replacement"

    with pytest.raises(ValidationError):
        _evidence(source_reference="")


@pytest.mark.parametrize(
    "status",
    [
        QueryStatus.SUCCESS_EMPTY,
        QueryStatus.PARTIAL,
        QueryStatus.DATA_UNAVAILABLE,
        QueryStatus.TOOL_ERROR,
        QueryStatus.TIMEOUT,
    ],
)
def test_failed_query_cannot_create_supporting_evidence(status: QueryStatus) -> None:
    with pytest.raises(ValidationError):
        _evidence(query_status=status, polarity=EvidencePolarity.SUPPORT)
