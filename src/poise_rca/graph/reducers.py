"""Serialized deterministic merge for parallel investigator results."""

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from poise_rca.agents.supervisor import HypothesisTransitionRecommendation
from poise_rca.domain import (
    CandidateProposal,
    CausalRole,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    InvestigationResult,
    QueryStatus,
    TaskStatus,
    validate_evidence_links,
)
from poise_rca.graph.state import DiagnosisState, StateUpdate


def _content_key(record: EvidenceRecord) -> str:
    content = {
        "modality": record.modality.value,
        "service": record.service,
        "observation": record.observation,
        "polarity": record.polarity.value,
        "anomaly": record.attributes.get("anomaly"),
    }
    return sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _merge_group(key: str, records: Sequence[EvidenceRecord]) -> EvidenceRecord:
    ordered = sorted(records, key=lambda item: (item.source_reference, item.evidence_id))
    first = ordered[0]
    sources = sorted({item.source_reference for item in ordered})
    query_ids = sorted({item.query_id for item in ordered})
    worker_ids = sorted({item.evidence_id for item in ordered})
    attributes = dict(first.attributes)
    attributes.update(
        {
            "source_references": sources,
            "query_ids": query_ids,
            "worker_evidence_ids": worker_ids,
        }
    )
    return first.model_copy(
        update={
            "evidence_id": str(uuid5(NAMESPACE_URL, f"poise-evidence:{key}")),
            "query_id": query_ids[0],
            "source_reference": sources[0],
            "parent_evidence_ids": tuple(worker_ids),
            "attributes": attributes,
            "created_at": min(item.created_at for item in ordered),
        }
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _proposal_identity(proposal: CandidateProposal) -> tuple[str, str, str]:
    return (
        _normalize_text(proposal.service),
        _normalize_text(proposal.fault_type),
        _normalize_text(proposal.causal_claim),
    )


def _hypothesis_identity(hypothesis: Hypothesis) -> tuple[str, str, str]:
    return (
        _normalize_text(hypothesis.root_cause_service),
        _normalize_text(hypothesis.fault_type),
        _normalize_text(hypothesis.causal_claim),
    )


def _evidence_connects_service(record: EvidenceRecord, service: str) -> bool:
    expected = service.casefold()
    linked_services = {
        record.service.casefold(),
        str(record.attributes.get("source_service", "")).casefold(),
        str(record.attributes.get("target_service", "")).casefold(),
    }
    return expected in linked_services


def validate_candidate_proposal(
    proposal: CandidateProposal,
    service_inventory: frozenset[str],
    evidence: dict[str, EvidenceRecord],
    hypotheses: dict[str, Hypothesis],
) -> Hypothesis:
    """Return a new or enriched ACTIVE candidate, or reject it with a reason."""

    missing = set(proposal.evidence_ids) - set(evidence)
    if missing:
        raise ValueError(f"proposal cites unknown evidence: {sorted(missing)}")
    safe_services = {service.casefold() for service in service_inventory}
    if proposal.service.casefold() not in safe_services:
        raise ValueError("proposal service is absent from safe inventory or observed telemetry")
    cited = [evidence[identifier] for identifier in proposal.evidence_ids]
    if not any(_evidence_connects_service(record, proposal.service) for record in cited):
        raise ValueError("proposal evidence does not connect the service to the incident")

    identity = _proposal_identity(proposal)
    existing = next(
        (item for item in hypotheses.values() if _hypothesis_identity(item) == identity),
        None,
    )
    if existing is not None:
        support = tuple(sorted(set(existing.supporting_evidence_ids) | set(proposal.evidence_ids)))
        return existing.model_copy(update={"supporting_evidence_ids": support})

    hypothesis_id = str(uuid5(NAMESPACE_URL, f"poise-rca:candidate:{'|'.join(identity)}"))
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        root_cause_service=proposal.service,
        fault_type=proposal.fault_type,
        causal_claim=proposal.causal_claim,
        causal_role=CausalRole.UNKNOWN,
        supporting_evidence_ids=proposal.evidence_ids,
        status=HypothesisStatus.ACTIVE,
    )


def _merge_worker_evidence(
    state: DiagnosisState,
    results: Sequence[InvestigationResult],
) -> tuple[dict[str, EvidenceRecord], dict[str, str], set[str]]:
    merged = dict(state.evidence)
    content_to_existing = {
        _content_key(record): identifier for identifier, record in merged.items()
    }
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    evidence_id_keys: dict[str, set[str]] = defaultdict(set)
    for result in results:
        for record in result.evidence:
            key = _content_key(record)
            grouped[key].append(record)
            evidence_id_keys[record.evidence_id].add(key)

    conflicts = {
        f"conflicting-worker-evidence-id:{identifier}"
        for identifier, keys in evidence_id_keys.items()
        if len(keys) > 1
    }
    worker_to_merged: dict[str, str] = {}
    for key in sorted(grouped):
        records = grouped[key]
        existing_id = content_to_existing.get(key)
        if existing_id is None:
            canonical = _merge_group(key, records)
        else:
            novel_records = [record for record in records if record.evidence_id != existing_id]
            if novel_records:
                combined = _merge_group(key, [merged[existing_id], *novel_records])
                canonical = combined.model_copy(
                    update={
                        "evidence_id": existing_id,
                        "parent_evidence_ids": tuple(
                            identifier
                            for identifier in combined.parent_evidence_ids
                            if identifier != existing_id
                        ),
                    }
                )
            else:
                canonical = merged[existing_id]
        merged[canonical.evidence_id] = canonical
        content_to_existing[key] = canonical.evidence_id
        for record in records:
            worker_to_merged[record.evidence_id] = canonical.evidence_id
    return merged, worker_to_merged, conflicts


def _map_proposal(
    proposal: CandidateProposal,
    worker_to_merged: dict[str, str],
) -> CandidateProposal:
    return proposal.model_copy(
        update={
            "evidence_ids": tuple(
                worker_to_merged.get(identifier, identifier) for identifier in proposal.evidence_ids
            )
        }
    )


def _normalize_worker_result(
    result: InvestigationResult,
    merged_evidence: dict[str, EvidenceRecord],
    worker_to_merged: dict[str, str],
) -> InvestigationResult:
    mapped_ids = {
        worker_to_merged[record.evidence_id]
        for record in result.evidence
        if record.evidence_id in worker_to_merged
    }
    return result.model_copy(
        update={
            "evidence": tuple(merged_evidence[identifier] for identifier in sorted(mapped_ids)),
            "cited_evidence_ids": tuple(
                worker_to_merged.get(identifier, identifier)
                for identifier in result.cited_evidence_ids
            ),
            "candidate_proposals": tuple(
                _map_proposal(proposal, worker_to_merged) for proposal in result.candidate_proposals
            ),
        }
    )


def merge_worker_results(
    state: DiagnosisState,
    results: Sequence[InvestigationResult],
) -> StateUpdate:
    """Merge evidence first, then validate conclusions and dynamic candidates."""

    merged_evidence, worker_to_merged, merge_conflicts = _merge_worker_evidence(state, results)
    normalized_results = tuple(
        _normalize_worker_result(result, merged_evidence, worker_to_merged) for result in results
    )
    inventory = frozenset(
        set(state.service_inventory) | {record.service for record in merged_evidence.values()}
    )
    hypotheses = dict(state.hypotheses)
    tasks = dict(state.tasks)
    pending = list(state.pending_candidate_proposals)
    rejections = list(state.candidate_rejections)
    invalid_conclusions = set(state.invalid_conclusion_task_ids)
    failed_signatures = set(state.failed_query_signatures)
    query_statuses = dict(state.query_statuses)
    conflicts = set(state.conflicts) | merge_conflicts

    relationships: dict[tuple[str, str], set[EvidencePolarity]] = defaultdict(set)
    for record in merged_evidence.values():
        relationships[(record.modality.value, record.service)].add(record.polarity)
    for (modality, service), polarities in relationships.items():
        if {EvidencePolarity.SUPPORT, EvidencePolarity.CONTRADICT} <= polarities:
            conflict_hash = sha256(f"{modality}:{service}:support:contradict".encode()).hexdigest()[
                :12
            ]
            conflicts.add(f"conflict:{modality}:{service}:{conflict_hash}")

    for result in normalized_results:
        known_ids = set(merged_evidence)
        unknown_citations = set(result.cited_evidence_ids) - known_ids
        if result.conclusion is None or not result.cited_evidence_ids or unknown_citations:
            invalid_conclusions.add(result.task_id)
        task = tasks.get(result.task_id)
        if task is not None:
            tasks[result.task_id] = task.model_copy(
                update={
                    "status": (
                        TaskStatus.FAILED
                        if result.task_id in invalid_conclusions
                        else TaskStatus.COMPLETED
                    )
                }
            )
        result_statuses = getattr(result, "query_statuses", {})
        for signature, status in result_statuses.items():
            query_statuses[signature] = status
            if status in {QueryStatus.TOOL_ERROR, QueryStatus.TIMEOUT}:
                failed_signatures.add(signature)

        for proposal in result.candidate_proposals:
            try:
                candidate = validate_candidate_proposal(
                    proposal,
                    inventory,
                    merged_evidence,
                    hypotheses,
                )
            except ValueError as exc:
                rejections.append(f"{result.task_id}:{proposal.service}: {exc}")
                continue
            hypotheses[candidate.hypothesis_id] = candidate
            pending.append(proposal)

    validate_evidence_links(hypotheses, merged_evidence)
    return StateUpdate(
        service_inventory=inventory,
        evidence=merged_evidence,
        hypotheses=hypotheses,
        tasks=tasks,
        results=state.results + normalized_results,
        pending_candidate_proposals=tuple(pending),
        candidate_rejections=tuple(rejections),
        invalid_conclusion_task_ids=frozenset(invalid_conclusions),
        failed_query_signatures=frozenset(failed_signatures),
        query_statuses=query_statuses,
        conflicts=tuple(sorted(conflicts)),
        supervisor_steps=state.supervisor_steps,
        max_supervisor_steps=state.max_supervisor_steps,
    )


_TRANSITION_TARGETS: dict[HypothesisStatus, frozenset[HypothesisStatus]] = {
    HypothesisStatus.ACTIVE: frozenset(
        {HypothesisStatus.LEADING, HypothesisStatus.DEFERRED, HypothesisStatus.REFUTED}
    ),
    HypothesisStatus.LEADING: frozenset(
        {HypothesisStatus.ACTIVE, HypothesisStatus.DEFERRED, HypothesisStatus.REFUTED}
    ),
    HypothesisStatus.DEFERRED: frozenset({HypothesisStatus.ACTIVE, HypothesisStatus.REFUTED}),
    HypothesisStatus.REFUTED: frozenset({HypothesisStatus.ACTIVE}),
}

_BLOCKING_QUERY_STATUSES = {
    QueryStatus.SUCCESS_EMPTY,
    QueryStatus.PARTIAL,
    QueryStatus.DATA_UNAVAILABLE,
    QueryStatus.TOOL_ERROR,
    QueryStatus.TIMEOUT,
}


def _has_comparative_causal_support(records: Sequence[EvidenceRecord]) -> bool:
    return any(
        record.polarity is EvidencePolarity.SUPPORT
        and record.query_status is QueryStatus.SUCCESS_WITH_DATA
        and record.attributes.get("causal_support") is True
        and record.attributes.get("comparative") is True
        for record in records
    )


def _has_adequate_contradiction(records: Sequence[EvidenceRecord]) -> bool:
    return any(
        record.polarity is EvidencePolarity.CONTRADICT
        and record.query_status is QueryStatus.SUCCESS_WITH_DATA
        and record.attributes.get("necessary_prediction") is True
        and record.attributes.get("adequate_coverage") is True
        for record in records
    )


def _transition_guard_passes(
    source: HypothesisStatus,
    target: HypothesisStatus,
    records: Sequence[EvidenceRecord],
    query_statuses: Mapping[str, QueryStatus],
    credible_new_competitor: bool,
) -> bool:
    if target is HypothesisStatus.LEADING:
        return source is HypothesisStatus.ACTIVE and _has_comparative_causal_support(records)
    if target is HypothesisStatus.REFUTED:
        return _has_adequate_contradiction(records)
    if target is HypothesisStatus.DEFERRED:
        return any(status in _BLOCKING_QUERY_STATUSES for status in query_statuses.values())
    if source is HypothesisStatus.DEFERRED and target is HypothesisStatus.ACTIVE:
        return credible_new_competitor or any(
            record.query_status is QueryStatus.SUCCESS_WITH_DATA for record in records
        )
    if source is HypothesisStatus.LEADING and target is HypothesisStatus.ACTIVE:
        return credible_new_competitor or any(
            record.polarity is EvidencePolarity.CONTRADICT for record in records
        )
    if source is HypothesisStatus.REFUTED and target is HypothesisStatus.ACTIVE:
        return any(
            record.attributes.get("contradiction_invalidated") is True
            or record.attributes.get("provenance_invalid") is True
            for record in records
        )
    return False


def apply_hypothesis_transitions(
    hypotheses: Mapping[str, Hypothesis],
    evidence: Mapping[str, EvidenceRecord],
    recommendations: Sequence[HypothesisTransitionRecommendation],
    *,
    query_statuses: Mapping[str, QueryStatus] | None = None,
    credible_new_competitor: bool = False,
) -> dict[str, Hypothesis]:
    """Apply only explicit, evidence-grounded lifecycle changes with deterministic guards."""

    updated = dict(hypotheses)
    recommendation_counts: dict[str, int] = defaultdict(int)
    for recommendation in recommendations:
        recommendation_counts[recommendation.hypothesis_id] += 1

    for recommendation in sorted(
        recommendations,
        key=lambda item: (item.hypothesis_id, item.target_status.value),
    ):
        if recommendation_counts[recommendation.hypothesis_id] != 1:
            continue
        hypothesis = updated.get(recommendation.hypothesis_id)
        if hypothesis is None:
            continue
        target = recommendation.target_status
        if target not in _TRANSITION_TARGETS[hypothesis.status]:
            continue
        if not set(recommendation.evidence_ids) <= set(evidence):
            continue
        records = [evidence[identifier] for identifier in recommendation.evidence_ids]
        if not _transition_guard_passes(
            hypothesis.status,
            target,
            records,
            query_statuses or {},
            credible_new_competitor,
        ):
            continue

        changes: dict[str, object] = {"status": target}
        if target is HypothesisStatus.LEADING:
            changes["supporting_evidence_ids"] = tuple(
                sorted(set(hypothesis.supporting_evidence_ids) | set(recommendation.evidence_ids))
            )
            for identifier, other in tuple(updated.items()):
                if (
                    identifier != hypothesis.hypothesis_id
                    and other.status is HypothesisStatus.LEADING
                ):
                    updated[identifier] = other.model_copy(
                        update={"status": HypothesisStatus.ACTIVE}
                    )
        elif target is HypothesisStatus.REFUTED:
            changes["contradicting_evidence_ids"] = tuple(
                sorted(
                    set(hypothesis.contradicting_evidence_ids) | set(recommendation.evidence_ids)
                )
            )
        elif hypothesis.status is HypothesisStatus.REFUTED:
            changes["contradicting_evidence_ids"] = tuple(
                identifier
                for identifier in hypothesis.contradicting_evidence_ids
                if identifier not in recommendation.evidence_ids
            )
        updated[hypothesis.hypothesis_id] = hypothesis.model_copy(update=changes)

    validate_evidence_links(updated, evidence)
    return updated
