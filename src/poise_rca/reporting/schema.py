"""Machine-readable graded diagnosis report schema."""

from pydantic import BaseModel, ConfigDict, Field

from poise_rca.domain import CausalRole, DiagnosisStatus, HypothesisStatus


class RunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    code_sha: str
    config_fingerprint: str
    latency_ms: float = Field(ge=0)
    token_usage: int = Field(default=0, ge=0)
    model_cost: float = Field(default=0.0, ge=0)
    trace_id: str | None = None


class EvidenceCitation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    source_reference: str


class FactualClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str
    evidence_ids: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...]


class CandidateSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis_id: str
    service: str
    mechanism: str
    causal_claim: str
    causal_role: CausalRole
    status: HypothesisStatus
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]


class WorkerConclusionSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    conclusion: str
    cited_evidence_ids: tuple[str, ...]
    unresolved_questions: tuple[str, ...] = ()


class DiagnosisReport(BaseModel):
    """Terminal artifact containing facts, uncertainty, provenance, and run metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    case_id: str
    code_sha: str
    config_fingerprint: str
    status: DiagnosisStatus
    stop_reason: str
    confirmed_facts: tuple[FactualClaim, ...]
    candidates: tuple[CandidateSummary, ...]
    worker_conclusions: tuple[WorkerConclusionSummary, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    candidate_promotions: tuple[str, ...] = ()
    candidate_rejections: tuple[str, ...] = ()
    supervisor_steps: int = Field(ge=0)
    root_cause: CandidateSummary | None = None
    remaining_candidates: tuple[CandidateSummary, ...] = ()
    missing_or_unavailable_evidence: tuple[str, ...] = ()
    next_human_action: str | None = None
    tool_status_summary: dict[str, int]
    latency_ms: float
    token_usage: int
    model_cost: float
    trace_id: str | None = None
