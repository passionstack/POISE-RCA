"""LangGraph subgraph for one isolated, multi-step investigation worker."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from poise_rca.agents.hypothesis_generator import EvidencePromptSummary
from poise_rca.agents.worker import (
    WorkerAction,
    WorkerModel,
    WorkerPromptInput,
    WorkerQuerySummary,
)
from poise_rca.domain import (
    EvidenceRecord,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
)
from poise_rca.graph.dependencies import ToolExecutor


class WorkerState(TypedDict, total=False):
    """Private sequential state for a single worker invocation."""

    prompt_input: WorkerPromptInput
    available_evidence: dict[str, EvidenceRecord]
    local_evidence_ids: tuple[str, ...]
    query_history: tuple[WorkerQuerySummary, ...]
    query_signatures: tuple[str, ...]
    next_action: WorkerAction
    result: InvestigationResult


def create_worker_state(
    prompt_input: WorkerPromptInput,
    evidence: Mapping[str, EvidenceRecord],
) -> WorkerState:
    """Create fresh local state and discard evidence outside the worker projection."""

    allowed_ids = {item.evidence_id for item in prompt_input.evidence}
    return WorkerState(
        prompt_input=prompt_input,
        available_evidence={
            identifier: record
            for identifier, record in evidence.items()
            if identifier in allowed_ids
        },
        local_evidence_ids=(),
        query_history=(),
        query_signatures=(),
    )


def query_signature(action: WorkerAction) -> str:
    """Canonicalize a worker query independently of dictionary insertion order."""

    if action.action != "QUERY" or action.tool_name is None:
        raise ValueError("query signatures require a query action")
    arguments = json.dumps(action.arguments, sort_keys=True, separators=(",", ":"), default=str)
    return f"{action.tool_name.casefold()}|{arguments}"


def _prompt_for_step(state: WorkerState) -> WorkerPromptInput:
    base = state["prompt_input"]
    evidence = tuple(
        _evidence_summary(item)
        for item in sorted(
            state["available_evidence"].values(),
            key=lambda record: record.evidence_id,
        )
    )
    return base.model_copy(
        update={
            "evidence": evidence,
            "query_history": state.get("query_history", ()),
        }
    )


def _decide_action(state: WorkerState, model: WorkerModel) -> WorkerState:
    action = WorkerAction.model_validate(model.invoke(_prompt_for_step(state)))
    return WorkerState(next_action=action)


def _query_task(prompt_input: WorkerPromptInput, action: WorkerAction) -> InvestigationTask:
    assert action.tool_name is not None
    arguments = dict(action.arguments)
    first_service = prompt_input.hypotheses[0].root_cause_service
    modality_by_tool = {
        "query_logs": Modality.LOGS,
        "query_metrics": Modality.METRICS,
        "query_traces": Modality.TRACES,
    }
    task_data = prompt_input.task.model_dump()
    task_data.update(
        {
            "tool_name": action.tool_name,
            "arguments": arguments,
            "modality": modality_by_tool.get(action.tool_name),
            "service": arguments.get("service", first_service),
            "start_time": arguments.get("start_time", prompt_input.task.start_time),
            "end_time": arguments.get("end_time", prompt_input.task.end_time),
        }
    )
    return InvestigationTask.model_validate(task_data)


def _execute_query(state: WorkerState, tools: ToolExecutor) -> WorkerState:
    action = state["next_action"]
    allowed_tools = {item.name for item in state["prompt_input"].tools}
    if action.tool_name not in allowed_tools:
        raise ValueError(f"worker requested an unavailable tool: {action.tool_name}")
    signature = query_signature(action)
    history = state.get("query_history", ())
    signatures = state.get("query_signatures", ())
    if signature in signatures:
        duplicate = WorkerQuerySummary(
            signature=signature,
            tool_name=action.tool_name or "unknown",
            status="DUPLICATE",
            duplicate=True,
        )
        return WorkerState(query_history=(*history, duplicate))

    query_task = _query_task(state["prompt_input"], action)
    query_result = tools.execute(query_task)
    if query_result.task_id != state["prompt_input"].task.task_id:
        raise ValueError("tool result task ID does not match the assigned worker task")
    available = dict(state["available_evidence"])
    local_ids = list(state.get("local_evidence_ids", ()))
    for record in query_result.evidence:
        existing = available.get(record.evidence_id)
        if existing is not None and existing != record:
            raise ValueError(f"conflicting evidence record: {record.evidence_id}")
        available[record.evidence_id] = record
        if record.evidence_id not in local_ids:
            local_ids.append(record.evidence_id)
    status = query_result.status or QueryStatus.TOOL_ERROR
    summary = WorkerQuerySummary(
        signature=signature,
        tool_name=action.tool_name or "unknown",
        status=status.value,
        evidence_ids=tuple(record.evidence_id for record in query_result.evidence),
        error=query_result.error,
    )
    return WorkerState(
        available_evidence=available,
        local_evidence_ids=tuple(local_ids),
        query_history=(*history, summary),
        query_signatures=(*signatures, signature),
    )


def _compose_result(state: WorkerState) -> WorkerState:
    action = state["next_action"]
    if action.action != "REPORT" or action.conclusion is None:
        raise ValueError("worker result requires a report action")
    available = state["available_evidence"]
    available_ids = set(available)
    unknown_citations = set(action.cited_evidence_ids) - available_ids
    if unknown_citations:
        raise ValueError(f"worker report cites unknown evidence: {sorted(unknown_citations)}")
    if not action.cited_evidence_ids:
        raise ValueError("worker report must cite evidence")
    for proposal in action.candidate_proposals:
        missing = set(proposal.evidence_ids) - available_ids
        if missing:
            raise ValueError(f"candidate proposal cites unknown evidence: {sorted(missing)}")
    query_statuses: dict[str, QueryStatus] = {}
    for query in state.get("query_history", ()):
        if query.duplicate:
            continue
        try:
            query_statuses[query.signature] = QueryStatus(query.status)
        except ValueError:
            continue
    result = InvestigationResult(
        task_id=state["prompt_input"].task.task_id,
        evidence=tuple(available[identifier] for identifier in sorted(available)),
        conclusion=action.conclusion,
        cited_evidence_ids=action.cited_evidence_ids,
        candidate_proposals=action.candidate_proposals,
        unresolved_questions=action.unresolved_questions,
        query_signatures=state.get("query_signatures", ()),
        query_statuses=query_statuses,
    )
    return WorkerState(result=result)


def _route_action(state: WorkerState) -> str:
    return "execute_query" if state["next_action"].action == "QUERY" else "compose_result"


def build_worker_graph(
    model: WorkerModel,
    tools: ToolExecutor,
) -> CompiledStateGraph[WorkerState, None, WorkerState, WorkerState]:
    """Compile a fresh-context worker loop around injected model and tool boundaries."""

    graph = StateGraph(WorkerState)
    graph.add_node("decide_action", lambda state: _decide_action(state, model))
    graph.add_node("execute_query", lambda state: _execute_query(state, tools))
    graph.add_node("compose_result", _compose_result)
    graph.add_edge(START, "decide_action")
    graph.add_conditional_edges(
        "decide_action",
        _route_action,
        ["execute_query", "compose_result"],
    )
    graph.add_edge("execute_query", "decide_action")
    graph.add_edge("compose_result", END)
    return graph.compile()


def _evidence_summary(record: EvidenceRecord) -> EvidencePromptSummary:
    return EvidencePromptSummary(
        evidence_id=record.evidence_id,
        modality=record.modality.value,
        service=record.service,
        observation=record.observation,
        query_status=record.query_status.value,
    )
