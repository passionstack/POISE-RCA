from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from poise_rca.agents.worker import (
    PromptedWorkerModel,
    WorkerAction,
    WorkerPromptInput,
    WorkerToolDescription,
    build_worker_prompt_input,
)
from poise_rca.domain import (
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    InvestigationTask,
    Modality,
    QueryStatus,
    TaskKind,
)
from poise_rca.topology import build_observed_service_graph


class CapturingRunnable:
    def __init__(self, output: object) -> None:
        self.output = output
        self.inputs: list[object] = []

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        return self.output


def _evidence(service: str, observation: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.TRACES,
        service=service,
        observation=observation,
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference=f"C:/secret/raw/{service}.parquet",
        created_at=datetime.now(UTC),
    )


def _hypothesis(service: str, evidence_id: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service=service,
        fault_type="delay",
        causal_claim=f"{service} initiated the latency increase.",
        origin_evidence_ids=(evidence_id,),
        expected_observations=(f"{service} latency rises first",),
    )


def _context() -> tuple[
    InvestigationTask,
    dict[str, Hypothesis],
    dict[str, EvidenceRecord],
]:
    checkout_evidence = _evidence("checkout", "Checkout latency rises first.")
    payment_evidence = _evidence("payment", "Payment errors rise near onset.")
    frontend_evidence = _evidence("frontend", "Frontend is slow downstream.")
    checkout = _hypothesis("checkout", checkout_evidence.evidence_id)
    payment = _hypothesis("payment", payment_evidence.evidence_id)
    frontend = _hypothesis("frontend", frontend_evidence.evidence_id)
    task = InvestigationTask(
        task_id=str(uuid4()),
        task_kind=TaskKind.COMPARISON,
        target_hypothesis_ids=(checkout.hypothesis_id, payment.hypothesis_id),
        question="Which service shows the initiating anomaly?",
        shared_evidence_ids=(checkout_evidence.evidence_id,),
        discriminating_observations=("relative anomaly onset",),
        available_modalities=(Modality.LOGS, Modality.TRACES),
    )
    return (
        task,
        {item.hypothesis_id: item for item in (checkout, payment, frontend)},
        {
            item.evidence_id: item
            for item in (checkout_evidence, payment_evidence, frontend_evidence)
        },
    )


def test_worker_prompt_is_isolated_to_assigned_hypotheses_and_relevant_evidence() -> None:
    task, hypotheses, evidence = _context()

    prompt = build_worker_prompt_input(
        task,
        hypotheses,
        evidence,
        service_inventory=frozenset({"checkout", "payment", "frontend"}),
        topology=build_observed_service_graph(tuple(evidence.values())),
        tools=(
            WorkerToolDescription(
                name="query_traces",
                description="Query normalized traces.",
                argument_schema={"type": "object"},
            ),
        ),
    )

    assert {item.hypothesis_id for item in prompt.hypotheses} == set(task.target_hypothesis_ids)
    assert {item.evidence_id for item in prompt.evidence} == {
        evidence_id
        for hypothesis_id in task.target_hypothesis_ids
        for evidence_id in hypotheses[hypothesis_id].origin_evidence_ids
    }
    assert "frontend" not in {item.root_cause_service for item in prompt.hypotheses}
    serialized = prompt.model_dump_json()
    assert "C:/secret/raw" not in serialized
    assert "ground_truth" not in serialized
    assert "messages" not in serialized


def test_worker_action_has_query_or_free_form_report_shapes() -> None:
    query = WorkerAction(
        action="QUERY",
        tool_name="query_traces",
        arguments={"service": "checkout"},
    )
    report = WorkerAction(
        action="REPORT",
        conclusion="Trace ordering currently favors checkout, while payment remains viable.",
        cited_evidence_ids=(str(uuid4()),),
        unresolved_questions=("Payment log coverage is unavailable.",),
    )

    assert query.action == "QUERY"
    assert report.action == "REPORT"
    assert not hasattr(report, "outcome")
    with pytest.raises(ValidationError):
        WorkerAction(action="REPORT", conclusion="   ")
    with pytest.raises(ValidationError):
        WorkerAction(action="QUERY", tool_name="query_traces", conclusion="premature")


def test_prompted_worker_model_uses_worker_prompt_and_structured_input() -> None:
    task, hypotheses, evidence = _context()
    prompt_input: WorkerPromptInput = build_worker_prompt_input(
        task,
        hypotheses,
        evidence,
        service_inventory=frozenset(evidence_item.service for evidence_item in evidence.values()),
        topology=build_observed_service_graph(tuple(evidence.values())),
        tools=(),
    )
    runnable = CapturingRunnable(
        {
            "action": "REPORT",
            "conclusion": "Both candidates remain viable.",
            "cited_evidence_ids": prompt_input.task.shared_evidence_ids,
        }
    )

    output = PromptedWorkerModel(runnable).invoke(prompt_input)

    assert isinstance(runnable.inputs[0][0], SystemMessage)  # type: ignore[index]
    assert isinstance(runnable.inputs[0][1], HumanMessage)  # type: ignore[index]
    system_message = runnable.inputs[0][0]  # type: ignore[index]
    normalized_system = " ".join(str(system_message.content).lower().split())
    assert "natural-language conclusion" in normalized_system
    assert output["action"] == "REPORT"  # type: ignore[index]
