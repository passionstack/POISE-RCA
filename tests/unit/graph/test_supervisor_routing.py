from datetime import UTC, datetime, timedelta
from uuid import uuid4

from langgraph.types import Send

from poise_rca.agents import (
    HypothesisBatch,
    HypothesisPromptInput,
    InvestigationTaskDraft,
    SupervisorPlan,
    SupervisorPromptInput,
    WorkerPromptInput,
)
from poise_rca.domain import (
    CandidateProposal,
    CaseContext,
    DiagnosisStatus,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    TaskKind,
    TaskStatus,
)
from poise_rca.graph.dependencies import GraphDependencies
from poise_rca.graph.nodes import supervisor_plan_node, validate_plan_node
from poise_rca.graph.routers import route_after_evaluation, route_validated_tasks
from poise_rca.graph.state import DiagnosisState
from poise_rca.graph.stopping import DiagnosisDecision
from poise_rca.topology import build_observed_service_graph


class UnusedHypothesisModel:
    def invoke(self, input: HypothesisPromptInput) -> object:
        del input
        return HypothesisBatch(hypotheses=())


class CapturingSupervisor:
    def __init__(self, plan: SupervisorPlan) -> None:
        self.plan = plan
        self.inputs: list[SupervisorPromptInput] = []

    def invoke(self, input: SupervisorPromptInput) -> object:
        self.inputs.append(input)
        return self.plan.model_dump()


class UnusedWorkerModel:
    def invoke(self, input: WorkerPromptInput) -> object:
        raise AssertionError(f"worker should not run in routing test: {input.task.task_id}")


class UnusedTools:
    def execute(self, task: InvestigationTask) -> InvestigationResult:
        raise AssertionError(f"tool should not run in routing test: {task.task_id}")


def _evidence(service: str = "checkout") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.TRACES,
        service=service,
        observation=f"{service} latency rises.",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference="trace:test",
        created_at=datetime.now(UTC),
    )


def _hypothesis(service: str, evidence_id: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service=service,
        fault_type="delay",
        causal_claim=f"{service} delay initiated the incident.",
        origin_evidence_ids=(evidence_id,),
        expected_observations=(f"{service} anomaly occurs first",),
    )


def _state(*hypotheses: Hypothesis, evidence: EvidenceRecord) -> DiagnosisState:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return DiagnosisState(
        case=CaseContext(
            case_id="re2-ob-0000000000000001",
            incident_start=start,
            incident_end=start + timedelta(minutes=5),
            initial_symptom="Elevated latency.",
            available_modalities=frozenset({Modality.TRACES}),
        ),
        service_inventory=frozenset({item.root_cause_service for item in hypotheses}),
        topology=build_observed_service_graph((evidence,)),
        evidence={evidence.evidence_id: evidence},
        hypotheses={item.hypothesis_id: item for item in hypotheses},
        tasks={},
        results=(),
        conflicts=(),
    )


def _deps(plan: SupervisorPlan) -> GraphDependencies:
    return GraphDependencies(
        hypothesis_model=UnusedHypothesisModel(),
        supervisor_model=CapturingSupervisor(plan),
        worker_model=UnusedWorkerModel(),
        tool_registry=UnusedTools(),
    )


def _draft(
    kind: TaskKind,
    targets: tuple[str, ...],
    evidence_id: str,
) -> InvestigationTaskDraft:
    return InvestigationTaskDraft(
        task_kind=kind,
        target_hypothesis_ids=targets,
        question="Which candidate is the initiating cause?",
        shared_evidence_ids=(evidence_id,),
        discriminating_observations=("relative trace onset",),
        available_modalities=(Modality.TRACES,),
    )


def test_validated_tasks_fan_out_with_fresh_isolated_worker_inputs() -> None:
    evidence = _evidence()
    first = _hypothesis("checkout", evidence.evidence_id)
    second = _hypothesis("payment", evidence.evidence_id)
    third = _hypothesis("frontend", evidence.evidence_id)
    drafts = (
        _draft(
            TaskKind.COMPARISON,
            (first.hypothesis_id, second.hypothesis_id),
            evidence.evidence_id,
        ),
        _draft(
            TaskKind.COMPARISON,
            (second.hypothesis_id, third.hypothesis_id),
            evidence.evidence_id,
        ),
    )
    state = _state(first, second, third, evidence=evidence).model_dump(mode="python")
    deps = _deps(SupervisorPlan(tasks=drafts))

    planned = supervisor_plan_node(state, deps)
    state.update(planned)
    validated = validate_plan_node(state, deps)
    state.update(validated)
    routes = route_validated_tasks(state)

    assert planned["supervisor_steps"] == 1
    assert isinstance(routes, list)
    assert len(routes) == 2
    assert all(isinstance(route, Send) for route in routes)
    worker_inputs = state["worker_inputs"]
    assert len({id(item) for item in worker_inputs.values()}) == 2
    for task_id, prompt in worker_inputs.items():
        assert {item.hypothesis_id for item in prompt.hypotheses} == set(
            state["tasks"][task_id].target_hypothesis_ids
        )


def test_single_other_unknown_candidate_uses_challenge_not_fake_comparison() -> None:
    evidence = _evidence()
    other = _hypothesis("OTHER/UNKNOWN", evidence.evidence_id)
    draft = _draft(TaskKind.CHALLENGE, (other.hypothesis_id,), evidence.evidence_id)
    state = _state(other, evidence=evidence).model_dump(mode="python")
    deps = _deps(SupervisorPlan(tasks=(draft,)))

    state.update(supervisor_plan_node(state, deps))
    state.update(validate_plan_node(state, deps))

    task = next(iter(state["tasks"].values()))
    assert task.task_kind is TaskKind.CHALLENGE
    assert task.target_hypothesis_ids == (other.hypothesis_id,)


def test_repeated_completed_plan_increments_step_without_redispatch() -> None:
    evidence = _evidence()
    left = _hypothesis("checkout", evidence.evidence_id)
    right = _hypothesis("payment", evidence.evidence_id)
    draft = _draft(
        TaskKind.COMPARISON,
        (left.hypothesis_id, right.hypothesis_id),
        evidence.evidence_id,
    )
    state = _state(left, right, evidence=evidence).model_dump(mode="python")
    deps = _deps(SupervisorPlan(tasks=(draft,)))
    state.update(supervisor_plan_node(state, deps))
    state.update(validate_plan_node(state, deps))
    task = next(iter(state["tasks"].values()))
    state["tasks"] = {task.task_id: task.model_copy(update={"status": TaskStatus.COMPLETED})}
    state["worker_inputs"] = {}

    state.update(supervisor_plan_node(state, deps))
    state.update(validate_plan_node(state, deps))

    assert state["supervisor_steps"] == 2
    assert route_validated_tasks(state) == "audit_evidence"


def test_pending_candidate_proposal_routes_back_to_supervisor_and_blocks_report() -> None:
    evidence = _evidence("fraud")
    hypothesis = _hypothesis("checkout", evidence.evidence_id)
    state = _state(hypothesis, evidence=evidence).model_dump(mode="python")
    state["pending_candidate_proposals"] = (
        CandidateProposal(
            service="fraud",
            fault_type="delay",
            causal_claim="Fraud delay may initiate the incident.",
            evidence_ids=(evidence.evidence_id,),
            suggested_investigation="Compare fraud and checkout.",
        ),
    )
    state["decision"] = DiagnosisDecision(
        status=DiagnosisStatus.CONFIRMED,
        stop_reason="A leader exists.",
        root_cause_hypothesis_id=hypothesis.hypothesis_id,
    )
    state["supervisor_steps"] = 1
    state["max_supervisor_steps"] = 10

    assert route_after_evaluation(state) == "supervisor_plan"
