"""Read-only claim-to-evidence audit boundary."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from poise_rca.domain import HypothesisStatus, validate_evidence_links
from poise_rca.graph.state import DiagnosisState


class ClaimCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str
    evidence_ids: tuple[str, ...]
    grounded: bool


class AuditResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claims: tuple[ClaimCheck, ...]
    unresolved_competitor_warnings: tuple[str, ...] = ()


class AuditModel(Protocol):
    def invoke(self, input: dict[str, object]) -> object:
        """Return a value parseable as AuditResult."""


def deterministic_audit(state: DiagnosisState) -> AuditResult:
    """Audit ledger facts without asking a model to interpret raw telemetry."""

    validate_evidence_links(state.hypotheses, state.evidence)
    factual_claims = tuple(
        ClaimCheck(
            claim=record.observation,
            evidence_ids=(record.evidence_id,),
            grounded=True,
        )
        for record in sorted(state.evidence.values(), key=lambda item: item.evidence_id)
    )
    hypothesis_claims = tuple(
        ClaimCheck(
            claim=hypothesis.causal_claim,
            evidence_ids=hypothesis.supporting_evidence_ids,
            grounded=bool(hypothesis.supporting_evidence_ids),
        )
        for hypothesis in sorted(state.hypotheses.values(), key=lambda item: item.hypothesis_id)
    )
    unresolved = tuple(
        f"{hypothesis.hypothesis_id}: {hypothesis.root_cause_service} remains unresolved"
        for hypothesis in sorted(state.hypotheses.values(), key=lambda item: item.hypothesis_id)
        if hypothesis.status is not HypothesisStatus.REFUTED
    )
    return AuditResult(
        claims=factual_claims + hypothesis_claims,
        unresolved_competitor_warnings=unresolved,
    )


def audit_claims(state: DiagnosisState, model: AuditModel) -> AuditResult:
    """Audit existing claims and references without allowing evidence creation."""

    validate_evidence_links(state.hypotheses, state.evidence)
    payload: dict[str, object] = {
        "hypotheses": [item.model_dump() for item in state.hypotheses.values()],
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "observation": item.observation,
                "modality": item.modality.value,
                "service": item.service,
            }
            for item in state.evidence.values()
        ],
    }
    result = AuditResult.model_validate(model.invoke(payload))
    known_ids = set(state.evidence)
    if any(not set(check.evidence_ids) <= known_ids for check in result.claims):
        raise ValueError("audit result contains an unknown evidence ID")
    return result
