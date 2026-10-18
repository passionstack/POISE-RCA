from datetime import UTC, datetime, timedelta
from uuid import uuid4

from poise_rca.agents.supervisor import (
    HypothesisTransitionRecommendation,
    InvestigationTaskDraft,
    SupervisorPlan,
    SupervisorPromptInput,
    plan_investigations,
    task_signature,
    validate_supervisor_plan,
    validate_transition_recommendations,
)
from poise_rca.domain import (
    CausalRole,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    Modality,
    QueryStatus,
    TaskKind,
)
from poise_rca.topology import build_observed_service_graph


class CapturingSupervisorModel:
    def __init__(self) -> None:
        self.inputs: list[SupervisorPromptInput] = []

    def invoke(self, input: SupervisorPromptInput) -> object:
        self.inputs.append(input)
        return {"tasks": [], "no_meaningful_competition": False}


def _evidence() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.TRACES,
        service="checkout",
        observation="Checkout latency rises before frontend latency.",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference="trace:checkout->frontend",
        created_at=datetime.now(UTC),
    )


def _hypothesis(
    service: str,
    evidence_id: str,
    *,
    role: CausalRole = CausalRole.UNKNOWN,
    status: HypothesisStatus = HypothesisStatus.ACTIVE,
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service=service,
        fault_type="delay",
        causal_claim=f"{service} delay explains the incident.",
        causal_role=role,
        origin_evidence_ids=(evidence_id,),
        contradicting_evidence_ids=(evidence_id,) if status is HypothesisStatus.REFUTED else (),
        expected_observations=(f"{service} anomaly precedes propagated symptoms",),
        status=status,
    )


def _draft(kind: TaskKind, targets: tuple[str, ...], evidence_id: str) -> InvestigationTaskDraft:
    return InvestigationTaskDraft(
        task_kind=kind,
        target_hypothesis_ids=targets,
        question="Which hypothesis best explains the first observed anomaly?",
        shared_evidence_ids=(evidence_id,),
        discriminating_observations=("relative anomaly onset",),
        available_modalities=(Modality.TRACES,),
    )


def _validate(
    plan: SupervisorPlan,
    hypotheses: dict[str, Hypothesis],
    evidence: EvidenceRecord,
    *,
    completed: frozenset[str] = frozenset(),
) -> tuple:
    return validate_supervisor_plan(
        plan,
        hypotheses,
        {evidence.evidence_id: evidence},
        completed_task_signatures=completed,
        available_modalities=frozenset({Modality.TRACES}),
        incident_start=evidence.created_at,
        incident_end=evidence.created_at + timedelta(minutes=5),
    )


def test_comparison_accepts_two_distinct_non_refuted_hypotheses() -> None:
    evidence = _evidence()
    left = _hypothesis("checkout", evidence.evidence_id)
    right = _hypothesis("payment", evidence.evidence_id)
    draft = _draft(
        TaskKind.COMPARISON,
        (right.hypothesis_id, left.hypothesis_id),
        evidence.evidence_id,
    )

    tasks = _validate(
        SupervisorPlan(tasks=(draft,)),
        {left.hypothesis_id: left, right.hypothesis_id: right},
        evidence,
    )

    assert len(tasks) == 1
    assert tasks[0].task_kind is TaskKind.COMPARISON
    assert tasks[0].target_hypothesis_ids == (right.hypothesis_id, left.hypothesis_id)
    assert tasks[0].start_time == evidence.created_at
    assert tasks[0].end_time == evidence.created_at + timedelta(minutes=5)


def test_comparison_rejects_wrong_shape_refuted_and_other_unknown() -> None:
    evidence = _evidence()
    active = _hypothesis("checkout", evidence.evidence_id)
    refuted = _hypothesis("payment", evidence.evidence_id, status=HypothesisStatus.REFUTED)
    other = _hypothesis("OTHER/UNKNOWN", evidence.evidence_id)
    hypotheses = {item.hypothesis_id: item for item in (active, refuted, other)}
    drafts = (
        _draft(TaskKind.COMPARISON, (active.hypothesis_id,), evidence.evidence_id),
        _draft(
            TaskKind.COMPARISON,
            (active.hypothesis_id, refuted.hypothesis_id),
            evidence.evidence_id,
        ),
        _draft(
            TaskKind.COMPARISON,
            (active.hypothesis_id, other.hypothesis_id),
            evidence.evidence_id,
        ),
    )

    assert _validate(SupervisorPlan(tasks=drafts), hypotheses, evidence) == ()


def test_challenge_accepts_exactly_one_hypothesis() -> None:
    evidence = _evidence()
    hypothesis = _hypothesis("checkout", evidence.evidence_id)

    tasks = _validate(
        SupervisorPlan(
            tasks=(
                _draft(
                    TaskKind.CHALLENGE,
                    (hypothesis.hypothesis_id,),
                    evidence.evidence_id,
                ),
            )
        ),
        {hypothesis.hypothesis_id: hypothesis},
        evidence,
    )

    assert len(tasks) == 1
    assert tasks[0].task_kind is TaskKind.CHALLENGE


def test_dangling_evidence_and_hypothesis_ids_are_rejected() -> None:
    evidence = _evidence()
    hypothesis = _hypothesis("checkout", evidence.evidence_id)
    draft = _draft(
        TaskKind.COMPARISON,
        (hypothesis.hypothesis_id, str(uuid4())),
        str(uuid4()),
    )

    assert (
        _validate(
            SupervisorPlan(tasks=(draft,)),
            {hypothesis.hypothesis_id: hypothesis},
            evidence,
        )
        == ()
    )


def test_compatible_causal_layers_require_initiating_cause_question() -> None:
    evidence = _evidence()
    cause = _hypothesis(
        "checkout",
        evidence.evidence_id,
        role=CausalRole.INITIATING_CAUSE,
    )
    symptom = _hypothesis(
        "frontend",
        evidence.evidence_id,
        role=CausalRole.PROPAGATED_SYMPTOM,
    )
    draft = _draft(
        TaskKind.COMPARISON,
        (cause.hypothesis_id, symptom.hypothesis_id),
        evidence.evidence_id,
    )
    hypotheses = {cause.hypothesis_id: cause, symptom.hypothesis_id: symptom}

    assert _validate(SupervisorPlan(tasks=(draft,)), hypotheses, evidence) == ()

    causal_draft = draft.model_copy(
        update={"question": "Which hypothesis represents the initiating cause?"}
    )
    assert len(_validate(SupervisorPlan(tasks=(causal_draft,)), hypotheses, evidence)) == 1


def test_duplicate_completed_signature_is_rejected_and_ids_are_stable() -> None:
    evidence = _evidence()
    left = _hypothesis("checkout", evidence.evidence_id)
    right = _hypothesis("payment", evidence.evidence_id)
    draft = _draft(
        TaskKind.COMPARISON,
        (left.hypothesis_id, right.hypothesis_id),
        evidence.evidence_id,
    )
    hypotheses = {left.hypothesis_id: left, right.hypothesis_id: right}
    first = _validate(SupervisorPlan(tasks=(draft,)), hypotheses, evidence)[0]
    reordered = draft.model_copy(
        update={
            "target_hypothesis_ids": (right.hypothesis_id, left.hypothesis_id),
            "question": "  Which hypothesis best explains the first observed anomaly?  ",
        }
    )
    second = _validate(SupervisorPlan(tasks=(reordered,)), hypotheses, evidence)[0]

    assert first.task_id == second.task_id
    assert (
        _validate(
            SupervisorPlan(tasks=(draft,)),
            hypotheses,
            evidence,
            completed=frozenset({task_signature(first)}),
        )
        == ()
    )


def test_transition_recommendation_requires_rationale_evidence_and_legal_transition() -> None:
    evidence = _evidence()
    hypothesis = _hypothesis("checkout", evidence.evidence_id)
    valid = HypothesisTransitionRecommendation(
        hypothesis_id=hypothesis.hypothesis_id,
        target_status=HypothesisStatus.LEADING,
        rationale="Trace ordering favors this candidate over its competitor.",
        evidence_ids=(evidence.evidence_id,),
    )
    dangling = valid.model_copy(update={"evidence_ids": (str(uuid4()),)})

    accepted = validate_transition_recommendations(
        (valid, dangling),
        {hypothesis.hypothesis_id: hypothesis},
        {evidence.evidence_id: evidence},
    )

    assert accepted == (valid,)
    assert hypothesis.status is HypothesisStatus.ACTIVE


def test_planning_uses_safe_structured_projections_without_mutating_state() -> None:
    evidence = _evidence()
    hypothesis = _hypothesis("checkout", evidence.evidence_id)
    model = CapturingSupervisorModel()

    plan = plan_investigations(
        {hypothesis.hypothesis_id: hypothesis},
        {evidence.evidence_id: evidence},
        completed_investigations=(),
        completed_task_signatures=(),
        pending_proposals=(),
        topology=build_observed_service_graph((evidence,)),
        available_modalities={Modality.TRACES},
        model=model,
    )

    assert plan.tasks == ()
    assert model.inputs[0].hypotheses[0].status is HypothesisStatus.ACTIVE
    assert "source_reference" not in model.inputs[0].model_dump_json()
    assert hypothesis.status is HypothesisStatus.ACTIVE
