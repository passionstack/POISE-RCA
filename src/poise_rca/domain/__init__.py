"""Public domain contracts shared by POISE-RCA subsystems."""

from poise_rca.domain.analysis import AnomalyFinding, StaticAnalysisResult
from poise_rca.domain.case import CaseContext
from poise_rca.domain.enums import (
    CausalRole,
    DiagnosisStatus,
    EvidencePolarity,
    HypothesisStatus,
    Modality,
    QueryStatus,
    TaskKind,
    TaskStatus,
)
from poise_rca.domain.evidence import EvidenceRecord
from poise_rca.domain.hypothesis import Hypothesis, validate_evidence_links
from poise_rca.domain.tasks import (
    CandidateProposal,
    InvestigationResult,
    InvestigationTask,
)

__all__ = [
    "AnomalyFinding",
    "CaseContext",
    "CandidateProposal",
    "CausalRole",
    "DiagnosisStatus",
    "EvidencePolarity",
    "EvidenceRecord",
    "Hypothesis",
    "HypothesisStatus",
    "InvestigationResult",
    "InvestigationTask",
    "Modality",
    "QueryStatus",
    "StaticAnalysisResult",
    "TaskStatus",
    "TaskKind",
    "validate_evidence_links",
]
