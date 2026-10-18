from datetime import UTC, datetime
from uuid import uuid4

import pytest

from poise_rca.agents.supervisor import HypothesisTransitionRecommendation
from poise_rca.domain import (
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    Modality,
    QueryStatus,
)
from poise_rca.graph.reducers import apply_hypothesis_transitions


def _evidence(
    *,
    polarity: EvidencePolarity = EvidencePolarity.NEUTRAL,
    status: QueryStatus = QueryStatus.SUCCESS_WITH_DATA,
    attributes: dict[str, object] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.TRACES,
        service="checkout",
        observation="A scoped trace observation.",
        query_id=str(uuid4()),
        query_status=status,
        polarity=polarity,
        source_reference="trace:test",
        attributes=attributes or {},
        created_at=datetime.now(UTC),
    )


def _hypothesis(status: HypothesisStatus, evidence_id: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service="checkout",
        fault_type="delay",
        causal_claim="Checkout delay initiated the incident.",
        origin_evidence_ids=(evidence_id,),
        contradicting_evidence_ids=(evidence_id,) if status is HypothesisStatus.REFUTED else (),
        expected_observations=("Checkout delay precedes frontend latency.",),
        status=status,
    )


def _recommendation(
    hypothesis: Hypothesis,
    target: HypothesisStatus,
    evidence_id: str,
) -> HypothesisTransitionRecommendation:
    return HypothesisTransitionRecommendation(
        hypothesis_id=hypothesis.hypothesis_id,
        target_status=target,
        rationale="The cited observation satisfies the deterministic transition guard.",
        evidence_ids=(evidence_id,),
    )


@pytest.mark.parametrize(
    ("source", "target", "evidence", "query_statuses", "credible_competitor"),
    (
        (
            HypothesisStatus.ACTIVE,
            HypothesisStatus.LEADING,
            _evidence(
                polarity=EvidencePolarity.SUPPORT,
                attributes={"causal_support": True, "comparative": True},
            ),
            {},
            False,
        ),
        (
            HypothesisStatus.ACTIVE,
            HypothesisStatus.DEFERRED,
            _evidence(status=QueryStatus.DATA_UNAVAILABLE),
            {"q": QueryStatus.DATA_UNAVAILABLE},
            False,
        ),
        (
            HypothesisStatus.ACTIVE,
            HypothesisStatus.REFUTED,
            _evidence(
                polarity=EvidencePolarity.CONTRADICT,
                attributes={"necessary_prediction": True, "adequate_coverage": True},
            ),
            {},
            False,
        ),
        (
            HypothesisStatus.LEADING,
            HypothesisStatus.ACTIVE,
            _evidence(),
            {},
            True,
        ),
        (
            HypothesisStatus.LEADING,
            HypothesisStatus.DEFERRED,
            _evidence(status=QueryStatus.SUCCESS_EMPTY),
            {"q": QueryStatus.SUCCESS_EMPTY},
            False,
        ),
        (
            HypothesisStatus.LEADING,
            HypothesisStatus.REFUTED,
            _evidence(
                polarity=EvidencePolarity.CONTRADICT,
                attributes={"necessary_prediction": True, "adequate_coverage": True},
            ),
            {},
            False,
        ),
        (
            HypothesisStatus.DEFERRED,
            HypothesisStatus.ACTIVE,
            _evidence(),
            {},
            False,
        ),
        (
            HypothesisStatus.DEFERRED,
            HypothesisStatus.REFUTED,
            _evidence(
                polarity=EvidencePolarity.CONTRADICT,
                attributes={"necessary_prediction": True, "adequate_coverage": True},
            ),
            {},
            False,
        ),
        (
            HypothesisStatus.REFUTED,
            HypothesisStatus.ACTIVE,
            _evidence(
                polarity=EvidencePolarity.CONTRADICT,
                attributes={"contradiction_invalidated": True},
            ),
            {},
            False,
        ),
    ),
)
def test_every_allowed_transition_requires_its_evidence_guard(
    source: HypothesisStatus,
    target: HypothesisStatus,
    evidence: EvidenceRecord,
    query_statuses: dict[str, QueryStatus],
    credible_competitor: bool,
) -> None:
    hypothesis = _hypothesis(source, evidence.evidence_id)

    updated = apply_hypothesis_transitions(
        {hypothesis.hypothesis_id: hypothesis},
        {evidence.evidence_id: evidence},
        (_recommendation(hypothesis, target, evidence.evidence_id),),
        query_statuses=query_statuses,
        credible_new_competitor=credible_competitor,
    )

    assert updated[hypothesis.hypothesis_id].status is target
    assert hypothesis.status is source


def test_missing_support_and_task_completion_cannot_refute_candidate() -> None:
    evidence = _evidence()
    hypothesis = _hypothesis(HypothesisStatus.ACTIVE, evidence.evidence_id)

    updated = apply_hypothesis_transitions(
        {hypothesis.hypothesis_id: hypothesis},
        {evidence.evidence_id: evidence},
        (
            _recommendation(
                hypothesis,
                HypothesisStatus.REFUTED,
                evidence.evidence_id,
            ),
        ),
    )
    unchanged = apply_hypothesis_transitions(
        {hypothesis.hypothesis_id: hypothesis},
        {evidence.evidence_id: evidence},
        (),
    )

    assert updated[hypothesis.hypothesis_id].status is HypothesisStatus.ACTIVE
    assert unchanged[hypothesis.hypothesis_id].status is HypothesisStatus.ACTIVE


def test_unavailable_or_empty_observation_defers_but_never_refutes() -> None:
    evidence = _evidence(status=QueryStatus.DATA_UNAVAILABLE)
    hypothesis = _hypothesis(HypothesisStatus.ACTIVE, evidence.evidence_id)
    statuses = {"trace-query": QueryStatus.DATA_UNAVAILABLE}

    refuted = apply_hypothesis_transitions(
        {hypothesis.hypothesis_id: hypothesis},
        {evidence.evidence_id: evidence},
        (_recommendation(hypothesis, HypothesisStatus.REFUTED, evidence.evidence_id),),
        query_statuses=statuses,
    )
    deferred = apply_hypothesis_transitions(
        {hypothesis.hypothesis_id: hypothesis},
        {evidence.evidence_id: evidence},
        (_recommendation(hypothesis, HypothesisStatus.DEFERRED, evidence.evidence_id),),
        query_statuses=statuses,
    )

    assert refuted[hypothesis.hypothesis_id].status is HypothesisStatus.ACTIVE
    assert deferred[hypothesis.hypothesis_id].status is HypothesisStatus.DEFERRED


def test_promoting_new_leader_demotes_previous_leader() -> None:
    evidence = _evidence(
        polarity=EvidencePolarity.SUPPORT,
        attributes={"causal_support": True, "comparative": True},
    )
    previous = _hypothesis(HypothesisStatus.LEADING, evidence.evidence_id)
    candidate = _hypothesis(HypothesisStatus.ACTIVE, evidence.evidence_id).model_copy(
        update={"root_cause_service": "payment", "causal_claim": "Payment initiated failure."}
    )

    updated = apply_hypothesis_transitions(
        {
            previous.hypothesis_id: previous,
            candidate.hypothesis_id: candidate,
        },
        {evidence.evidence_id: evidence},
        (
            _recommendation(
                candidate,
                HypothesisStatus.LEADING,
                evidence.evidence_id,
            ),
        ),
    )

    assert updated[candidate.hypothesis_id].status is HypothesisStatus.LEADING
    assert updated[previous.hypothesis_id].status is HypothesisStatus.ACTIVE


def test_refuted_candidate_cannot_reopen_without_invalidated_contradiction_audit() -> None:
    evidence = _evidence(polarity=EvidencePolarity.CONTRADICT)
    hypothesis = _hypothesis(HypothesisStatus.REFUTED, evidence.evidence_id)

    updated = apply_hypothesis_transitions(
        {hypothesis.hypothesis_id: hypothesis},
        {evidence.evidence_id: evidence},
        (_recommendation(hypothesis, HypothesisStatus.ACTIVE, evidence.evidence_id),),
    )

    assert updated[hypothesis.hypothesis_id].status is HypothesisStatus.REFUTED
