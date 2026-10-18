"""Conditional routing for supervisor planning and isolated worker fan-out."""

from langgraph.types import Send

from poise_rca.agents import SupervisorPlan, WorkerPromptInput
from poise_rca.domain import DiagnosisStatus, EvidenceRecord, InvestigationTask, TaskStatus
from poise_rca.graph.nodes import WorkflowState
from poise_rca.graph.stopping import DiagnosisDecision


def route_validated_tasks(state: WorkflowState) -> list[Send] | str:
    """Fan only newly validated pending tasks into fresh isolated worker states."""

    worker_inputs = state.get("worker_inputs", {})
    evidence = state.get("evidence", {})
    routes: list[Send] = []
    for task_id in sorted(worker_inputs):
        task = InvestigationTask.model_validate(state["tasks"][task_id])
        if task.status is not TaskStatus.PENDING:
            continue
        prompt = WorkerPromptInput.model_validate(worker_inputs[task_id])
        relevant_ids = {item.evidence_id for item in prompt.evidence}
        local_evidence = {
            identifier: EvidenceRecord.model_validate(record)
            for identifier, record in evidence.items()
            if identifier in relevant_ids
        }
        routes.append(
            Send(
                "run_worker",
                {
                    "worker_prompt": prompt,
                    "worker_evidence": local_evidence,
                },
            )
        )
    return routes or "audit_evidence"


def route_after_evaluation(state: WorkflowState) -> str:
    """Replan unresolved work unless semantics or the supervisor guard stop the run."""

    if state.get("pending_candidate_proposals"):
        return "supervisor_plan"
    steps = state.get("supervisor_steps", 0)
    max_steps = state.get("max_supervisor_steps", 10)
    if steps >= max_steps:
        return "generate_report"
    decision = DiagnosisDecision.model_validate(state["decision"])
    if decision.status is DiagnosisStatus.CONFIRMED:
        return "generate_report"
    plan = SupervisorPlan.model_validate(state.get("supervisor_plan", {}))
    has_pending_task = any(
        InvestigationTask.model_validate(task).status is TaskStatus.PENDING
        for task in state.get("tasks", {}).values()
    )
    if plan.no_meaningful_competition and not has_pending_task:
        return "generate_report"
    return "supervisor_plan"
