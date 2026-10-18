from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from poise_rca.domain import (
    CandidateProposal,
    CausalRole,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    TaskKind,
)


def _hypothesis(**overrides: object) -> Hypothesis:
    values: dict[str, object] = {
        "hypothesis_id": str(uuid4()),
        "root_cause_service": "checkout",
        "fault_type": "delay",
        "causal_claim": "Checkout latency propagates to the frontend.",
        "expected_observations": ("checkout spans start slowing before frontend spans",),
    }
    values.update(overrides)
    return Hypothesis.model_validate(values)


def test_hypothesis_requires_expected_or_missing_observation() -> None:
    with pytest.raises(ValidationError):
        _hypothesis(expected_observations=(), missing_evidence=())


def test_refuted_hypothesis_retains_contradicting_evidence() -> None:
    with pytest.raises(ValidationError):
        _hypothesis(status=HypothesisStatus.REFUTED, contradicting_evidence_ids=())

    contradiction_id = str(uuid4())
    hypothesis = _hypothesis(
        status=HypothesisStatus.REFUTED,
        contradicting_evidence_ids=(contradiction_id,),
    )
    assert hypothesis.contradicting_evidence_ids == (contradiction_id,)


def test_task_requires_target_hypothesis_and_question() -> None:
    start = datetime.now(UTC)
    values = {
        "task_id": str(uuid4()),
        "target_hypothesis_ids": (),
        "question": "",
        "modality": Modality.TRACES,
        "service": "checkout",
        "start_time": start,
        "end_time": start + timedelta(minutes=5),
        "tool_name": "query_traces",
        "expected_outcomes": ("checkout latency leads frontend latency",),
    }

    with pytest.raises(ValidationError):
        InvestigationTask.model_validate(values)


def test_hypothesis_lifecycle_contains_exactly_four_states() -> None:
    assert {status.value for status in HypothesisStatus} == {
        "ACTIVE",
        "DEFERRED",
        "REFUTED",
        "LEADING",
    }
    hypothesis = _hypothesis()
    assert hypothesis.status is HypothesisStatus.ACTIVE
    assert hypothesis.causal_role is CausalRole.UNKNOWN


def test_comparison_requires_two_distinct_hypotheses() -> None:
    left_id = str(uuid4())
    right_id = str(uuid4())
    task = InvestigationTask(
        task_id=str(uuid4()),
        task_kind=TaskKind.COMPARISON,
        target_hypothesis_ids=(left_id, right_id),
        question="Which hypothesis explains the first anomaly?",
        discriminating_observations=("relative anomaly onset",),
        available_modalities=(Modality.TRACES,),
    )
    assert task.target_hypothesis_ids == (left_id, right_id)

    with pytest.raises(ValidationError):
        InvestigationTask.model_validate(
            {
                **task.model_dump(),
                "target_hypothesis_ids": (left_id,),
            }
        )

    with pytest.raises(ValidationError):
        InvestigationTask.model_validate(
            {
                **task.model_dump(),
                "target_hypothesis_ids": (left_id, left_id),
            }
        )


def test_challenge_requires_one_hypothesis() -> None:
    target_id = str(uuid4())
    task = InvestigationTask(
        task_id=str(uuid4()),
        task_kind=TaskKind.CHALLENGE,
        target_hypothesis_ids=(target_id,),
        question="What observation could falsify this hypothesis?",
        discriminating_observations=("an earlier upstream anomaly",),
        available_modalities=(Modality.LOGS, Modality.TRACES),
    )
    assert task.target_hypothesis_ids == (target_id,)


def test_worker_conclusion_is_free_text_with_valid_evidence_citations() -> None:
    task_id = str(uuid4())
    evidence_id = str(uuid4())
    evidence = EvidenceRecord(
        evidence_id=evidence_id,
        modality=Modality.TRACES,
        service="checkout",
        observation="Checkout latency rises before frontend latency.",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference="trace:checkout->frontend",
        created_at=datetime.now(UTC),
    )
    result = InvestigationResult(
        task_id=task_id,
        evidence=(evidence,),
        conclusion="The observed ordering favors the left hypothesis.",
        cited_evidence_ids=(evidence_id,),
        candidate_proposals=(),
        unresolved_questions=(),
        query_signatures=("query_traces:checkout",),
    )
    assert result.conclusion.startswith("The observed ordering")

    with pytest.raises(ValidationError):
        InvestigationResult.model_validate({**result.model_dump(), "conclusion": "   "})

    with pytest.raises(ValidationError):
        InvestigationResult.model_validate(
            {
                **result.model_dump(),
                "cited_evidence_ids": (str(uuid4()),),
            }
        )

    with pytest.raises(ValidationError):
        InvestigationResult.model_validate({**result.model_dump(), "outcome": "FAVORS_LEFT"})


def test_candidate_proposal_requires_evidence_and_investigation() -> None:
    proposal = CandidateProposal(
        service="payment",
        fault_type="latency",
        causal_claim="Payment latency precedes checkout latency.",
        evidence_ids=(str(uuid4()),),
        suggested_investigation="Compare payment and checkout onset times.",
    )
    assert proposal.service == "payment"
