"""Protocol and provenance conversion for initialization-time static analysis."""

from datetime import datetime
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from poise_rca.data import TelemetryBundle
from poise_rca.domain import (
    AnomalyFinding,
    CaseContext,
    EvidencePolarity,
    EvidenceRecord,
    QueryStatus,
    StaticAnalysisResult,
)


class StaticAnalyzer(Protocol):
    """Injectable static analyzer that never receives dataset labels or paths."""

    def analyze(
        self,
        case: CaseContext,
        telemetry: TelemetryBundle,
        service_inventory: frozenset[str],
    ) -> StaticAnalysisResult: ...


def finding_to_evidence(
    finding: AnomalyFinding,
    created_at: datetime,
) -> EvidenceRecord:
    """Convert one successful finding into immutable initialization evidence."""

    query_id = str(uuid5(NAMESPACE_URL, f"poise-static-query:{finding.finding_id}"))
    return EvidenceRecord(
        evidence_id=finding.finding_id,
        modality=finding.modality,
        service=finding.service,
        observation=finding.observation,
        query_id=query_id,
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference=finding.source_reference,
        attributes={
            "analyzer": finding.analyzer,
            "finding_id": finding.finding_id,
            "rank": finding.rank,
            "score": finding.score,
            "signal": finding.signal,
            "stage": "initialization",
        },
        created_at=created_at,
    )
