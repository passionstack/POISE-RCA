from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from poise_rca.agents.worker import (
    WorkerAction,
    WorkerPromptInput,
    WorkerToolDescription,
    build_worker_prompt_input,
)
from poise_rca.domain import (
    CandidateProposal,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    TaskKind,
)
from poise_rca.graph.worker import build_worker_graph, create_worker_state
from poise_rca.topology import build_observed_service_graph


class SequenceWorkerModel:
    def __init__(self, actions: Sequence[WorkerAction]) -> None:
        self._actions = iter(actions)
        self.inputs: list[WorkerPromptInput] = []

    def invoke(self, input: WorkerPromptInput) -> object:
        self.inputs.append(input)
        return next(self._actions).model_dump()


class FakeToolExecutor:
    def __init__(
        self,
        results: dict[str, InvestigationResult],
    ) -> None:
        self.results = results
        self.tasks: list[InvestigationTask] = []

    def execute(self, task: InvestigationTask) -> InvestigationResult:
        self.tasks.append(task)
        return self.results[task.tool_name or ""]


def _evidence(
    service: str,
    modality: Modality,
    observation: str,
    *,
    evidence_id: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id or str(uuid4()),
        modality=modality,
        service=service,
        observation=observation,
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference=f"{modality.value}:local-test",
        created_at=datetime.now(UTC),
    )


def _hypothesis(service: str, evidence_id: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service=service,
        fault_type="delay",
        causal_claim=f"{service} delay initiated the incident.",
        origin_evidence_ids=(evidence_id,),
        expected_observations=(f"{service} anomaly occurs before downstream symptoms",),
    )


def _task_and_context(
    *,
    kind: TaskKind = TaskKind.COMPARISON,
) -> tuple[InvestigationTask, dict[str, Hypothesis], dict[str, EvidenceRecord]]:
    initial = _evidence("checkout", Modality.TRACES, "Checkout latency rises near onset.")
    left = _hypothesis("checkout", initial.evidence_id)
    right = _hypothesis("payment", initial.evidence_id)
    targets = (
        (left.hypothesis_id, right.hypothesis_id)
        if kind is TaskKind.COMPARISON
        else (left.hypothesis_id,)
    )
    task = InvestigationTask(
        task_id=str(uuid4()),
        task_kind=kind,
        target_hypothesis_ids=targets,
        question="Which candidate is the initiating cause?",
        shared_evidence_ids=(initial.evidence_id,),
        discriminating_observations=("relative onset",),
        available_modalities=(Modality.LOGS, Modality.TRACES),
    )
    hypotheses = {item.hypothesis_id: item for item in (left, right)}
    return task, hypotheses, {initial.evidence_id: initial}


def _query(tool_name: str) -> WorkerAction:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return WorkerAction(
        action="QUERY",
        tool_name=tool_name,
        arguments={
            "service": "checkout",
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(minutes=5)).isoformat(),
        },
    )


def _query_result(task_id: str, evidence: EvidenceRecord) -> InvestigationResult:
    return InvestigationResult(
        task_id=task_id,
        evidence=(evidence,),
        query_id=evidence.query_id,
        status=QueryStatus.SUCCESS_WITH_DATA,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )


def _prompt_and_state(
    task: InvestigationTask,
    hypotheses: dict[str, Hypothesis],
    evidence: dict[str, EvidenceRecord],
) -> tuple[WorkerPromptInput, dict[str, object]]:
    prompt = build_worker_prompt_input(
        task,
        hypotheses,
        evidence,
        service_inventory=frozenset({"checkout", "payment", "fraud"}),
        topology=build_observed_service_graph(tuple(evidence.values())),
        tools=tuple(
            WorkerToolDescription(
                name=f"query_{modality.value}",
                description=f"Query normalized {modality.value}.",
                argument_schema={"type": "object"},
            )
            for modality in task.available_modalities
        ),
    )
    return prompt, create_worker_state(prompt, evidence)


def test_worker_queries_logs_then_traces_and_returns_grounded_free_form_result() -> None:
    task, hypotheses, evidence = _task_and_context()
    log_evidence = _evidence("checkout", Modality.LOGS, "Timeout logs begin at 10:01.")
    trace_evidence = _evidence(
        "checkout", Modality.TRACES, "Checkout spans slow before payment spans."
    )
    report = WorkerAction(
        action="REPORT",
        conclusion=(
            "The observed onset favors checkout over payment, although shared failures "
            "leave payment viable as a contributor."
        ),
        cited_evidence_ids=(log_evidence.evidence_id, trace_evidence.evidence_id),
    )
    model = SequenceWorkerModel((_query("query_logs"), _query("query_traces"), report))
    tools = FakeToolExecutor(
        {
            "query_logs": _query_result(task.task_id, log_evidence),
            "query_traces": _query_result(task.task_id, trace_evidence),
        }
    )
    prompt, state = _prompt_and_state(task, hypotheses, evidence)

    output = build_worker_graph(model, tools).invoke(state)
    result = output["result"]

    assert [item.tool_name for item in tools.tasks] == ["query_logs", "query_traces"]
    assert {item.evidence_id for item in result.evidence} >= {
        log_evidence.evidence_id,
        trace_evidence.evidence_id,
    }
    assert result.conclusion == report.conclusion
    assert result.cited_evidence_ids == report.cited_evidence_ids
    assert len(result.query_signatures) == 2
    assert not hasattr(result, "messages")
    assert len(model.inputs[2].query_history) == 2
    assert prompt.query_history == ()


def test_worker_query_inherits_supervisor_validated_incident_window() -> None:
    task, hypotheses, evidence = _task_and_context(kind=TaskKind.CHALLENGE)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    task = task.model_copy(update={"start_time": start, "end_time": start + timedelta(minutes=5)})
    log_evidence = _evidence("checkout", Modality.LOGS, "Timeout logs are present.")
    query = WorkerAction(action="QUERY", tool_name="query_logs", arguments={})
    report = WorkerAction(
        action="REPORT",
        conclusion="The inherited incident window contains a checkout timeout signal.",
        cited_evidence_ids=(log_evidence.evidence_id,),
    )
    model = SequenceWorkerModel((query, report))
    tools = FakeToolExecutor({"query_logs": _query_result(task.task_id, log_evidence)})
    _, state = _prompt_and_state(task, hypotheses, evidence)

    build_worker_graph(model, tools).invoke(state)

    assert tools.tasks[0].service == "checkout"
    assert tools.tasks[0].start_time == start
    assert tools.tasks[0].end_time == start + timedelta(minutes=5)


def test_duplicate_query_signature_executes_only_once() -> None:
    task, hypotheses, evidence = _task_and_context()
    log_evidence = _evidence("checkout", Modality.LOGS, "Timeout logs present.")
    query = _query("query_logs")
    report = WorkerAction(
        action="REPORT",
        conclusion="The repeated log request adds no new distinction; both remain viable.",
        cited_evidence_ids=(next(iter(evidence)),),
    )
    model = SequenceWorkerModel((query, query, report))
    tools = FakeToolExecutor({"query_logs": _query_result(task.task_id, log_evidence)})
    _, state = _prompt_and_state(task, hypotheses, evidence)

    result = build_worker_graph(model, tools).invoke(state)["result"]

    assert len(tools.tasks) == 1
    assert len(result.query_signatures) == 1
    assert model.inputs[-1].query_history[-1].duplicate is True


def test_unavailable_data_remains_a_limitation_not_contradicting_evidence() -> None:
    task, hypotheses, evidence = _task_and_context()
    unavailable = InvestigationResult(
        task_id=task.task_id,
        query_id=str(uuid4()),
        status=QueryStatus.DATA_UNAVAILABLE,
        error="logs are not present for this service",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    report = WorkerAction(
        action="REPORT",
        conclusion="The candidates cannot be distinguished because log coverage is unavailable.",
        cited_evidence_ids=(next(iter(evidence)),),
        unresolved_questions=("Need checkout and payment logs for the incident window.",),
    )
    model = SequenceWorkerModel((_query("query_logs"), report))
    tools = FakeToolExecutor({"query_logs": unavailable})
    _, state = _prompt_and_state(task, hypotheses, evidence)

    output = build_worker_graph(model, tools).invoke(state)
    result = output["result"]

    assert result.unresolved_questions == report.unresolved_questions
    assert all(item.polarity is not EvidencePolarity.CONTRADICT for item in result.evidence)
    assert output["query_history"][0].status == QueryStatus.DATA_UNAVAILABLE.value
    assert QueryStatus.DATA_UNAVAILABLE in result.query_statuses.values()


def test_challenge_report_can_propose_evidence_grounded_third_service() -> None:
    task, hypotheses, evidence = _task_and_context(kind=TaskKind.CHALLENGE)
    fraud_evidence = _evidence(
        "fraud",
        Modality.TRACES,
        "Fraud service latency precedes checkout latency.",
    )
    proposal = CandidateProposal(
        service="fraud",
        fault_type="delay",
        causal_claim="Fraud service delay may initiate checkout latency.",
        evidence_ids=(fraud_evidence.evidence_id,),
        suggested_investigation="Compare fraud and checkout onset using traces.",
    )
    report = WorkerAction(
        action="REPORT",
        conclusion="The challenge found a plausible upstream fraud-service alternative.",
        cited_evidence_ids=(fraud_evidence.evidence_id,),
        candidate_proposals=(proposal,),
    )
    model = SequenceWorkerModel((_query("query_traces"), report))
    tools = FakeToolExecutor({"query_traces": _query_result(task.task_id, fraud_evidence)})
    _, state = _prompt_and_state(task, hypotheses, evidence)

    result = build_worker_graph(model, tools).invoke(state)["result"]

    assert result.candidate_proposals == (proposal,)
    assert result.conclusion == report.conclusion


def test_report_rejects_citations_outside_assigned_or_locally_collected_evidence() -> None:
    task, hypotheses, evidence = _task_and_context()
    report = WorkerAction(
        action="REPORT",
        conclusion="An unsupported conclusion.",
        cited_evidence_ids=(str(uuid4()),),
    )
    model = SequenceWorkerModel((report,))
    tools = FakeToolExecutor({})
    _, state = _prompt_and_state(task, hypotheses, evidence)

    with pytest.raises(ValueError, match="unknown evidence"):
        build_worker_graph(model, tools).invoke(state)
