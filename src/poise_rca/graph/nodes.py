"""Single-responsibility nodes for supervisor-worker diagnosis."""

import operator
from typing import Annotated, TypedDict, cast

from poise_rca.agents import (
    HypothesisBatch,
    StructuredOutputError,
    SupervisorPlan,
    WorkerPromptInput,
    build_worker_prompt_input,
    generate_initial_hypotheses,
    plan_investigations,
    task_signature,
    validate_generated_hypotheses,
    validate_supervisor_plan,
    validate_transition_recommendations,
)
from poise_rca.agents.evidence_auditor import (
    AuditResult,
    audit_claims,
    deterministic_audit,
)
from poise_rca.domain import (
    AnomalyFinding,
    CandidateProposal,
    CaseContext,
    EvidenceRecord,
    Hypothesis,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    StaticAnalysisResult,
    TaskStatus,
)
from poise_rca.graph.dependencies import GraphDependencies
from poise_rca.graph.reducers import (
    apply_hypothesis_transitions,
    merge_worker_results,
)
from poise_rca.graph.state import DiagnosisState
from poise_rca.graph.stopping import DiagnosisDecision, evaluate_stop
from poise_rca.graph.worker import build_worker_graph, create_worker_state
from poise_rca.reporting import DiagnosisReport, build_report
from poise_rca.topology import ObservedServiceGraph


class WorkflowState(TypedDict, total=False):
    case: CaseContext
    service_inventory: frozenset[str]
    topology: ObservedServiceGraph
    initial_findings: dict[str, AnomalyFinding]
    evidence: dict[str, EvidenceRecord]
    hypotheses: dict[str, Hypothesis]
    tasks: dict[str, InvestigationTask]
    results: tuple[InvestigationResult, ...]
    pending_candidate_proposals: tuple[CandidateProposal, ...]
    candidate_rejections: tuple[str, ...]
    invalid_conclusion_task_ids: frozenset[str]
    supervisor_steps: int
    max_supervisor_steps: int
    failed_query_signatures: frozenset[str]
    query_statuses: dict[str, QueryStatus]
    unavailable_tools: frozenset[str]
    conflicts: tuple[str, ...]
    supervisor_plan: SupervisorPlan
    worker_inputs: dict[str, WorkerPromptInput]
    worker_prompt: WorkerPromptInput
    worker_evidence: dict[str, EvidenceRecord]
    worker_results: Annotated[list[InvestigationResult], operator.add]
    audit: AuditResult
    decision: DiagnosisDecision
    report: DiagnosisReport


def _diagnosis_state(state: WorkflowState) -> DiagnosisState:
    fields = DiagnosisState.model_fields
    values = {key: value for key, value in state.items() if key in fields}
    return DiagnosisState.model_validate(values)


def generate_initial_hypotheses_node(
    state: WorkflowState,
    deps: GraphDependencies,
) -> WorkflowState:
    """Generate the initial ACTIVE candidate set from static findings."""

    current = _diagnosis_state(state)
    if current.hypotheses:
        return {}
    statuses = {
        modality: current.query_statuses.get(
            f"static|{modality.value}",
            QueryStatus.DATA_UNAVAILABLE,
        )
        for modality in Modality
    }
    analysis = StaticAnalysisResult(
        findings=tuple(
            sorted(current.initial_findings.values(), key=lambda finding: finding.finding_id)
        ),
        statuses=statuses,
    )
    try:
        hypotheses = generate_initial_hypotheses(
            current.case,
            current.topology,
            current.service_inventory,
            analysis,
            tuple(current.evidence.values()),
            deps.hypothesis_model,
        )
    except StructuredOutputError as exc:
        hypotheses = validate_generated_hypotheses(
            HypothesisBatch(hypotheses=()),
            current.case,
            current.topology,
            current.evidence,
            service_inventory=current.service_inventory,
        )
        deps.tracer.event(
            "hypotheses.degraded",
            {"reason": type(exc).__name__, "fallback": "OTHER/UNKNOWN"},
            observation_type="agent",
        )
    deps.tracer.event(
        "hypotheses.generated",
        {
            "count": len(hypotheses),
            "hypothesis_ids": sorted(hypotheses),
        },
        observation_type="agent",
    )
    return {"hypotheses": hypotheses}


def supervisor_plan_node(
    state: WorkflowState,
    deps: GraphDependencies,
) -> WorkflowState:
    """Ask the global supervisor for tasks and evidence-citing status advice."""

    if deps.supervisor_model is None:
        raise RuntimeError("a supervisor model is required for MAS diagnosis")
    current = _diagnosis_state(state)
    completed = {
        task_signature(task)
        for task in current.tasks.values()
        if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
    }
    plan = plan_investigations(
        current.hypotheses,
        current.evidence,
        current.results,
        completed,
        current.pending_candidate_proposals,
        current.topology,
        current.case.available_modalities,
        deps.supervisor_model,
    )
    steps = current.supervisor_steps + 1
    deps.tracer.event(
        "supervisor.planned",
        {"step": steps, "proposed_tasks": len(plan.tasks)},
        observation_type="agent",
    )
    return {
        "supervisor_plan": plan,
        "supervisor_steps": steps,
        "max_supervisor_steps": deps.max_supervisor_steps,
    }


def validate_plan_node(
    state: WorkflowState,
    deps: GraphDependencies,
) -> WorkflowState:
    """Commit grounded novel tasks and guarded lifecycle recommendations."""

    current = _diagnosis_state(state)
    plan = SupervisorPlan.model_validate(state["supervisor_plan"])
    completed = {
        task_signature(task)
        for task in current.tasks.values()
        if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
    }
    validated_tasks = validate_supervisor_plan(
        plan,
        current.hypotheses,
        current.evidence,
        completed_task_signatures=completed,
        available_modalities=current.case.available_modalities,
        incident_start=current.case.incident_start,
        incident_end=current.case.incident_end,
    )
    tasks = dict(current.tasks)
    tasks.update({task.task_id: task for task in validated_tasks})
    recommendations = validate_transition_recommendations(
        plan.transition_recommendations,
        current.hypotheses,
        current.evidence,
    )
    hypotheses = apply_hypothesis_transitions(
        current.hypotheses,
        current.evidence,
        recommendations,
        query_statuses=current.query_statuses,
        credible_new_competitor=bool(current.pending_candidate_proposals),
    )
    lifecycle_transitions = [
        {
            "hypothesis_id": identifier,
            "from": current.hypotheses[identifier].status.value,
            "to": hypothesis.status.value,
        }
        for identifier, hypothesis in sorted(hypotheses.items())
        if identifier in current.hypotheses
        and current.hypotheses[identifier].status is not hypothesis.status
    ]
    worker_inputs = {
        task.task_id: build_worker_prompt_input(
            task,
            hypotheses,
            current.evidence,
            current.service_inventory,
            current.topology,
            deps.worker_tools,
        )
        for task in validated_tasks
    }
    deps.tracer.event(
        "supervisor.validated",
        {
            "accepted_tasks": len(validated_tasks),
            "accepted_task_ids": [task.task_id for task in validated_tasks],
            "accepted_transitions": lifecycle_transitions,
        },
    )
    return {
        "tasks": tasks,
        "hypotheses": hypotheses,
        "worker_inputs": worker_inputs,
        "pending_candidate_proposals": (),
    }


def run_worker_node(state: WorkflowState, deps: GraphDependencies) -> WorkflowState:
    """Run one task in a fresh local subgraph and return only its final result."""

    if deps.worker_model is None:
        raise RuntimeError("a worker model is required for MAS diagnosis")
    prompt = WorkerPromptInput.model_validate(state["worker_prompt"])
    initial = create_worker_state(prompt, state["worker_evidence"])
    output = build_worker_graph(deps.worker_model, deps.tool_registry).invoke(
        initial,
        config={"recursion_limit": 50},
    )
    result = InvestigationResult.model_validate(output["result"])
    deps.tracer.event(
        "worker.completed",
        {
            "task_id": result.task_id,
            "queries": len(result.query_signatures),
            "evidence": len(result.evidence),
            "evidence_ids": [item.evidence_id for item in result.evidence],
        },
        observation_type="agent",
    )
    return {"worker_results": [result]}


def merge_worker_results_node(
    state: WorkflowState,
    deps: GraphDependencies,
) -> WorkflowState:
    """Serialize parallel worker outputs into the authoritative diagnosis state."""

    current = _diagnosis_state(state)
    known_task_ids = {result.task_id for result in current.results}
    incoming = tuple(
        result
        for raw in state.get("worker_results", [])
        if (result := InvestigationResult.model_validate(raw)).task_id not in known_task_ids
    )
    update = merge_worker_results(current, incoming)
    promoted = [
        {
            "hypothesis_id": identifier,
            "service": hypothesis.root_cause_service,
        }
        for identifier, hypothesis in sorted(update.hypotheses.items())
        if identifier not in current.hypotheses
    ]
    deps.tracer.event(
        "workers.merged",
        {
            "new_results": len(incoming),
            "evidence_count": len(update.evidence),
            "new_evidence_ids": sorted(set(update.evidence) - set(current.evidence)),
            "candidate_promotions": promoted,
            "candidate_rejections": list(
                update.candidate_rejections[len(current.candidate_rejections) :]
            ),
        },
    )
    return cast(
        WorkflowState,
        {
            "service_inventory": update.service_inventory,
            "evidence": update.evidence,
            "hypotheses": update.hypotheses,
            "tasks": update.tasks,
            "results": update.results,
            "pending_candidate_proposals": update.pending_candidate_proposals,
            "candidate_rejections": update.candidate_rejections,
            "invalid_conclusion_task_ids": update.invalid_conclusion_task_ids,
            "failed_query_signatures": update.failed_query_signatures,
            "query_statuses": update.query_statuses,
            "conflicts": update.conflicts,
            "supervisor_steps": current.supervisor_steps,
            "max_supervisor_steps": current.max_supervisor_steps,
        },
    )


def audit_evidence_node(state: WorkflowState, deps: GraphDependencies) -> WorkflowState:
    current = _diagnosis_state(state)
    audit = (
        audit_claims(current, deps.audit_model)
        if deps.audit_model is not None
        else deterministic_audit(current)
    )
    deps.tracer.event(
        "evidence.audited",
        {
            "claim_count": len(audit.claims),
            "grounded_count": sum(check.grounded for check in audit.claims),
        },
        observation_type="evaluator",
    )
    return {"audit": audit}


def evaluate_stop_node(state: WorkflowState, deps: GraphDependencies) -> WorkflowState:
    decision = evaluate_stop(_diagnosis_state(state))
    deps.tracer.event(
        "diagnosis.evaluated",
        {"status": decision.status.value, "stop_reason": decision.stop_reason},
        observation_type="evaluator",
    )
    return {"decision": decision}


def generate_report_node(state: WorkflowState, deps: GraphDependencies) -> WorkflowState:
    decision = DiagnosisDecision.model_validate(state["decision"])
    report = build_report(
        _diagnosis_state(state),
        deps.run_metadata,
        decision=decision,
    )
    deps.tracer.event(
        "report.generated",
        {
            "status": report.status.value,
            "run_id": report.run_id,
            "supervisor_steps": report.supervisor_steps,
            "hypothesis_states": {
                item.hypothesis_id: item.status.value for item in report.candidates
            },
            "candidate_promotions": list(report.candidate_promotions),
            "candidate_rejections": list(report.candidate_rejections),
        },
    )
    return {"report": report}
