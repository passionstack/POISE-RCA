"""LangGraph builder for supervisor-worker root-cause investigation."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from poise_rca.graph.dependencies import GraphDependencies
from poise_rca.graph.nodes import (
    WorkflowState,
    audit_evidence_node,
    evaluate_stop_node,
    generate_initial_hypotheses_node,
    generate_report_node,
    merge_worker_results_node,
    run_worker_node,
    supervisor_plan_node,
    validate_plan_node,
)
from poise_rca.graph.routers import route_after_evaluation, route_validated_tasks


def build_diagnosis_graph(
    deps: GraphDependencies,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[WorkflowState, None, WorkflowState, WorkflowState]:
    """Compile the global graph; every dispatched task invokes a fresh worker subgraph."""

    graph = StateGraph(WorkflowState)
    graph.add_node(
        "generate_initial_hypotheses",
        lambda state: generate_initial_hypotheses_node(state, deps),
    )
    graph.add_node("supervisor_plan", lambda state: supervisor_plan_node(state, deps))
    graph.add_node("validate_plan", lambda state: validate_plan_node(state, deps))
    graph.add_node("run_worker", lambda state: run_worker_node(state, deps))
    graph.add_node(
        "merge_worker_results",
        lambda state: merge_worker_results_node(state, deps),
    )
    graph.add_node("audit_evidence", lambda state: audit_evidence_node(state, deps))
    graph.add_node("evaluate_stop", lambda state: evaluate_stop_node(state, deps))
    graph.add_node("generate_report", lambda state: generate_report_node(state, deps))

    graph.add_edge(START, "generate_initial_hypotheses")
    graph.add_edge("generate_initial_hypotheses", "supervisor_plan")
    graph.add_edge("supervisor_plan", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        route_validated_tasks,
        ["run_worker", "audit_evidence"],
    )
    graph.add_edge("run_worker", "merge_worker_results")
    graph.add_edge("merge_worker_results", "audit_evidence")
    graph.add_edge("audit_evidence", "evaluate_stop")
    graph.add_conditional_edges(
        "evaluate_stop",
        route_after_evaluation,
        ["supervisor_plan", "generate_report"],
    )
    graph.add_edge("generate_report", END)
    return graph.compile(checkpointer=checkpointer)
