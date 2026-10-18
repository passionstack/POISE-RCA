"""Diagnosis report construction and atomic JSON persistence."""

import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

from poise_rca.domain import QueryStatus
from poise_rca.graph.state import DiagnosisState
from poise_rca.graph.stopping import DiagnosisDecision, evaluate_stop
from poise_rca.reporting.schema import (
    CandidateSummary,
    DiagnosisReport,
    EvidenceCitation,
    FactualClaim,
    RunMetadata,
    WorkerConclusionSummary,
)


def _candidate(state: DiagnosisState, hypothesis_id: str) -> CandidateSummary:
    hypothesis = state.hypotheses[hypothesis_id]
    return CandidateSummary(
        hypothesis_id=hypothesis.hypothesis_id,
        service=hypothesis.root_cause_service,
        mechanism=hypothesis.fault_type,
        causal_claim=hypothesis.causal_claim,
        causal_role=hypothesis.causal_role,
        status=hypothesis.status,
        supporting_evidence_ids=hypothesis.supporting_evidence_ids,
        contradicting_evidence_ids=hypothesis.contradicting_evidence_ids,
    )


def build_report(
    state: DiagnosisState,
    run_metadata: RunMetadata,
    decision: DiagnosisDecision | None = None,
) -> DiagnosisReport:
    """Build a report exclusively from validated diagnostic state."""

    decision = decision or evaluate_stop(state)
    facts = tuple(
        FactualClaim(
            claim=record.observation,
            evidence_ids=(record.evidence_id,),
            citations=(
                EvidenceCitation(
                    evidence_id=record.evidence_id,
                    source_reference=record.source_reference,
                ),
            ),
        )
        for record in sorted(state.evidence.values(), key=lambda item: item.evidence_id)
        if record.query_status is QueryStatus.SUCCESS_WITH_DATA
    )
    root_cause = (
        _candidate(state, decision.root_cause_hypothesis_id)
        if decision.root_cause_hypothesis_id is not None
        else None
    )
    remaining = tuple(
        _candidate(state, hypothesis_id)
        for hypothesis_id in decision.remaining_hypothesis_ids
        if hypothesis_id in state.hypotheses
    )
    unavailable = {
        f"query outcome: {status.value}"
        for status in state.query_statuses.values()
        if status
        in {
            QueryStatus.PARTIAL,
            QueryStatus.DATA_UNAVAILABLE,
            QueryStatus.TOOL_ERROR,
            QueryStatus.TIMEOUT,
        }
    }
    missing = tuple(sorted(set(decision.missing_evidence) | unavailable))
    status_counts = Counter(status.value for status in state.query_statuses.values())
    candidates = tuple(
        _candidate(state, hypothesis_id) for hypothesis_id in sorted(state.hypotheses)
    )
    valid_results = tuple(
        result
        for result in state.results
        if result.task_id not in state.invalid_conclusion_task_ids
        and result.conclusion is not None
        and set(result.cited_evidence_ids) <= set(state.evidence)
    )
    worker_conclusions = tuple(
        WorkerConclusionSummary(
            task_id=result.task_id,
            conclusion=result.conclusion or "",
            cited_evidence_ids=result.cited_evidence_ids,
            unresolved_questions=result.unresolved_questions,
        )
        for result in valid_results
    )
    retained_identities = {
        (
            hypothesis.root_cause_service.casefold(),
            hypothesis.fault_type.casefold(),
            " ".join(hypothesis.causal_claim.casefold().split()),
        )
        for hypothesis in state.hypotheses.values()
    }
    promotions = tuple(
        dict.fromkeys(
            proposal.service
            for result in valid_results
            for proposal in result.candidate_proposals
            if (
                proposal.service.casefold(),
                proposal.fault_type.casefold(),
                " ".join(proposal.causal_claim.casefold().split()),
            )
            in retained_identities
        )
    )
    return DiagnosisReport(
        run_id=run_metadata.run_id,
        case_id=state.case.case_id,
        code_sha=run_metadata.code_sha,
        config_fingerprint=run_metadata.config_fingerprint,
        status=decision.status,
        stop_reason=decision.stop_reason,
        confirmed_facts=facts,
        candidates=candidates,
        worker_conclusions=worker_conclusions,
        unresolved_questions=decision.unresolved_questions,
        candidate_promotions=promotions,
        candidate_rejections=state.candidate_rejections,
        supervisor_steps=state.supervisor_steps,
        root_cause=root_cause,
        remaining_candidates=remaining,
        missing_or_unavailable_evidence=tuple(sorted(set(missing) | set(decision.limitations))),
        next_human_action=decision.next_human_action,
        tool_status_summary=dict(sorted(status_counts.items())),
        latency_ms=run_metadata.latency_ms,
        token_usage=run_metadata.token_usage,
        model_cost=run_metadata.model_cost,
        trace_id=run_metadata.trace_id,
    )


def write_report(report: DiagnosisReport, path: Path) -> None:
    """Atomically write UTF-8 JSON through a temporary sibling file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(f"{report.model_dump_json(indent=2)}\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
