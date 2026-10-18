"""Grounded structured hypothesis generation with one schema-repair attempt."""

from collections.abc import Mapping, Sequence
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, ValidationError

from poise_rca.domain import (
    AnomalyFinding,
    CaseContext,
    CausalRole,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    Modality,
    QueryStatus,
    StaticAnalysisResult,
)
from poise_rca.topology import ObservedServiceGraph

_FAULT_TYPES = {"cpu", "mem", "disk", "delay", "loss", "socket", "unknown"}


class EvidencePromptSummary(BaseModel):
    """Evidence projection that intentionally excludes raw paths and rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    modality: str
    service: str
    observation: str
    query_status: str


class FindingPromptSummary(BaseModel):
    """Ranked anomaly projection without source paths or dataset labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str
    service: str
    modality: Modality
    signal: str
    observation: str
    analyzer: str
    score: float | None = None
    rank: int | None = None


class HypothesisPromptInput(BaseModel):
    """Only label-safe data supplied to a hypothesis model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case: CaseContext
    topology: ObservedServiceGraph
    service_inventory: frozenset[str]
    findings: tuple[FindingPromptSummary, ...]
    analysis_statuses: dict[Modality, QueryStatus]
    evidence: tuple[EvidencePromptSummary, ...]
    repair_error: str | None = None


class HypothesisDraft(BaseModel):
    """Structured candidate authored by the model before grounding validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_cause_service: str
    fault_type: str
    causal_claim: str
    causal_role: CausalRole = CausalRole.UNKNOWN
    origin_evidence_ids: tuple[str, ...] = ()
    expected_observations: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()


class HypothesisBatch(BaseModel):
    """Schema expected from the structured model invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypotheses: tuple[HypothesisDraft, ...]


class HypothesisModel(Protocol):
    def invoke(self, input: HypothesisPromptInput) -> object:
        """Return an object parseable as HypothesisBatch."""


class StructuredOutputError(RuntimeError):
    """Raised after both the initial and repair outputs violate the schema."""


def _hypothesis_id(draft: HypothesisDraft) -> str:
    identity = (
        f"{draft.root_cause_service.casefold()}:{draft.fault_type.casefold()}:"
        f"{draft.causal_claim.strip()}"
    )
    return str(uuid5(NAMESPACE_URL, f"poise-rca:hypothesis:{identity}"))


def _other_unknown() -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid5(NAMESPACE_URL, "poise-rca:hypothesis:other-unknown")),
        root_cause_service="OTHER/UNKNOWN",
        fault_type="unknown",
        causal_claim="An unobserved service or mechanism may explain the incident.",
        causal_role=CausalRole.UNKNOWN,
        missing_evidence=(
            "Additional independent telemetry is required to exclude unobserved causes.",
        ),
        status=HypothesisStatus.ACTIVE,
    )


def validate_generated_hypotheses(
    batch: HypothesisBatch,
    case: CaseContext,
    topology: ObservedServiceGraph,
    evidence: Mapping[str, EvidenceRecord],
    *,
    service_inventory: frozenset[str] | None = None,
) -> dict[str, Hypothesis]:
    """Drop invented references and retain an unbounded grounded candidate set."""

    allowed_services = set(service_inventory or ())
    allowed_services.update(node.service for node in topology.nodes)
    if case.entry_service is not None:
        allowed_services.add(case.entry_service)
    allowed_evidence = set(evidence)
    accepted: list[Hypothesis] = []
    for draft in batch.hypotheses:
        if draft.root_cause_service not in allowed_services:
            continue
        if draft.fault_type.casefold() not in _FAULT_TYPES:
            continue
        if not draft.origin_evidence_ids:
            continue
        if not set(draft.origin_evidence_ids) <= allowed_evidence:
            continue
        if not draft.expected_observations and not draft.missing_evidence:
            continue
        try:
            accepted.append(
                Hypothesis(
                    hypothesis_id=_hypothesis_id(draft),
                    root_cause_service=draft.root_cause_service,
                    fault_type=draft.fault_type.casefold(),
                    causal_claim=draft.causal_claim,
                    causal_role=draft.causal_role,
                    origin_evidence_ids=draft.origin_evidence_ids,
                    expected_observations=draft.expected_observations,
                    missing_evidence=draft.missing_evidence,
                    status=HypothesisStatus.ACTIVE,
                )
            )
        except ValidationError:
            continue

    accepted.append(_other_unknown())
    hypothesis_ids = {item.hypothesis_id for item in accepted}
    return {
        item.hypothesis_id: item.model_copy(
            update={
                "competing_hypothesis_ids": tuple(sorted(hypothesis_ids - {item.hypothesis_id}))
            }
        )
        for item in accepted
    }


def _invoke_model(
    prompt_input: HypothesisPromptInput,
    model: HypothesisModel,
) -> HypothesisBatch:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            return HypothesisBatch.model_validate(model.invoke(prompt_input))
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                detail = f"{type(exc).__name__}: {exc}"[:1000]
                prompt_input = prompt_input.model_copy(update={"repair_error": detail})
    raise StructuredOutputError(
        "hypothesis model returned invalid structured output"
    ) from last_error


def generate_initial_hypotheses(
    case: CaseContext,
    topology: ObservedServiceGraph,
    service_inventory: frozenset[str],
    analysis: StaticAnalysisResult,
    evidence: Sequence[EvidenceRecord],
    model: HypothesisModel,
) -> dict[str, Hypothesis]:
    """Generate any number of grounded ACTIVE candidates from static findings."""

    prompt_input = HypothesisPromptInput(
        case=case,
        topology=topology,
        service_inventory=service_inventory,
        findings=tuple(_finding_summary(finding) for finding in analysis.findings),
        analysis_statuses=analysis.statuses,
        evidence=tuple(_evidence_summary(record) for record in evidence),
    )
    batch = _invoke_model(prompt_input, model)
    evidence_by_id = {record.evidence_id: record for record in evidence}
    return validate_generated_hypotheses(
        batch,
        case,
        topology,
        evidence_by_id,
        service_inventory=service_inventory,
    )


def _finding_summary(finding: AnomalyFinding) -> FindingPromptSummary:
    return FindingPromptSummary(
        finding_id=finding.finding_id,
        service=finding.service,
        modality=finding.modality,
        signal=finding.signal,
        observation=finding.observation,
        analyzer=finding.analyzer,
        score=finding.score,
        rank=finding.rank,
    )


def _evidence_summary(record: EvidenceRecord) -> EvidencePromptSummary:
    return EvidencePromptSummary(
        evidence_id=record.evidence_id,
        modality=record.modality.value,
        service=record.service,
        observation=record.observation,
        query_status=record.query_status.value,
    )
