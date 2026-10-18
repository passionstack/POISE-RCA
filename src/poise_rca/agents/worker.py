"""Isolated worker prompt contracts and structured model adapter."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self

import yaml  # type: ignore[import-untyped]
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from poise_rca.agents.hypothesis_generator import EvidencePromptSummary
from poise_rca.domain import (
    CandidateProposal,
    CausalRole,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    InvestigationTask,
)
from poise_rca.topology import (
    ObservedServiceGraph,
    TopologyCompleteness,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class WorkerHypothesisView(BaseModel):
    """One assigned candidate visible to an isolated worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis_id: str
    root_cause_service: str
    fault_type: str
    causal_claim: str
    causal_role: CausalRole
    status: HypothesisStatus
    expected_observations: tuple[str, ...]
    missing_evidence: tuple[str, ...]


class WorkerToolDescription(BaseModel):
    """Safe tool schema exposed to a worker without an executable object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: NonEmptyStr
    description: NonEmptyStr
    argument_schema: dict[str, Any] = Field(default_factory=dict)


class WorkerQuerySummary(BaseModel):
    """Structured feedback from one local telemetry query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signature: str
    tool_name: str
    status: str
    evidence_ids: tuple[str, ...] = ()
    error: str | None = None
    duplicate: bool = False


class WorkerPromptInput(BaseModel):
    """Fresh context for one model step in a single investigation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: InvestigationTask
    hypotheses: tuple[WorkerHypothesisView, ...]
    evidence: tuple[EvidencePromptSummary, ...]
    service_inventory: frozenset[str]
    topology: ObservedServiceGraph
    tools: tuple[WorkerToolDescription, ...]
    query_history: tuple[WorkerQuerySummary, ...] = ()


class WorkerAction(BaseModel):
    """One query request or final evidence-grounded natural-language report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["QUERY", "REPORT"]
    tool_name: NonEmptyStr | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    conclusion: NonEmptyStr | None = None
    cited_evidence_ids: tuple[str, ...] = ()
    candidate_proposals: tuple[CandidateProposal, ...] = ()
    unresolved_questions: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def action_shape_is_valid(self) -> Self:
        if self.action == "QUERY":
            if self.tool_name is None:
                raise ValueError("a query action requires a tool name")
            if (
                self.conclusion is not None
                or self.cited_evidence_ids
                or self.candidate_proposals
                or self.unresolved_questions
            ):
                raise ValueError("a query action cannot contain report fields")
        elif self.tool_name is not None or self.arguments:
            raise ValueError("a report action cannot contain query fields")
        elif self.conclusion is None:
            raise ValueError("a report action requires a conclusion")
        return self


class WorkerModel(Protocol):
    """Structured model boundary for an isolated worker step."""

    def invoke(self, input: WorkerPromptInput) -> object:
        """Return an object parseable as a worker action."""


class MessageRunnable(Protocol):
    """Minimal LangChain-compatible runnable used by the prompt adapter."""

    def invoke(self, input: object) -> object:
        """Invoke a structured model with messages."""


class PromptedWorkerModel:
    """Render each worker step as an independent structured invocation."""

    def __init__(self, runnable: MessageRunnable) -> None:
        self._runnable = runnable
        path = Path(__file__).with_name("prompts") / "worker.yaml"
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping) or not isinstance(loaded.get("system"), str):
            raise ValueError("worker prompt requires system text")
        self._system_prompt = loaded["system"].strip()

    def invoke(self, input: WorkerPromptInput) -> object:
        messages: list[BaseMessage] = [SystemMessage(content=self._system_prompt)]
        payload = input.model_dump_json(indent=2)
        messages.append(HumanMessage(content=f"Investigation input:\n{payload}"))
        return self._runnable.invoke(messages)


def build_worker_prompt_input(
    task: InvestigationTask,
    hypotheses: Mapping[str, Hypothesis],
    evidence: Mapping[str, EvidenceRecord],
    service_inventory: frozenset[str],
    topology: ObservedServiceGraph,
    tools: Sequence[WorkerToolDescription],
) -> WorkerPromptInput:
    """Build a fresh worker context containing only assigned diagnostic material."""

    assigned = tuple(hypotheses[identifier] for identifier in task.target_hypothesis_ids)
    relevant_evidence_ids = set(task.shared_evidence_ids)
    for hypothesis in assigned:
        relevant_evidence_ids.update(hypothesis.origin_evidence_ids)
        relevant_evidence_ids.update(hypothesis.supporting_evidence_ids)
        relevant_evidence_ids.update(hypothesis.contradicting_evidence_ids)
    relevant_evidence = tuple(
        _evidence_summary(evidence[identifier])
        for identifier in sorted(relevant_evidence_ids)
        if identifier in evidence
    )
    relevant_services = {item.root_cause_service for item in assigned}
    return WorkerPromptInput(
        task=task,
        hypotheses=tuple(_hypothesis_view(item) for item in assigned),
        evidence=relevant_evidence,
        service_inventory=service_inventory,
        topology=_topology_slice(topology, relevant_services),
        tools=tuple(tools),
    )


def _hypothesis_view(hypothesis: Hypothesis) -> WorkerHypothesisView:
    return WorkerHypothesisView(
        hypothesis_id=hypothesis.hypothesis_id,
        root_cause_service=hypothesis.root_cause_service,
        fault_type=hypothesis.fault_type,
        causal_claim=hypothesis.causal_claim,
        causal_role=hypothesis.causal_role,
        status=hypothesis.status,
        expected_observations=hypothesis.expected_observations,
        missing_evidence=hypothesis.missing_evidence,
    )


def _evidence_summary(record: EvidenceRecord) -> EvidencePromptSummary:
    return EvidencePromptSummary(
        evidence_id=record.evidence_id,
        modality=record.modality.value,
        service=record.service,
        observation=record.observation,
        query_status=record.query_status.value,
    )


def _topology_slice(
    topology: ObservedServiceGraph,
    relevant_services: set[str],
) -> ObservedServiceGraph:
    edges = tuple(
        edge
        for edge in topology.edges
        if edge.source in relevant_services or edge.target in relevant_services
    )
    included_services = relevant_services | {
        endpoint for edge in edges for endpoint in (edge.source, edge.target)
    }
    nodes = tuple(node for node in topology.nodes if node.service in included_services)
    return ObservedServiceGraph(
        nodes=nodes,
        edges=edges,
        completeness=TopologyCompleteness(
            trace_evidence_count=topology.completeness.trace_evidence_count,
            observed_edge_count=len(edges),
            trace_coverage_complete=topology.completeness.trace_coverage_complete,
        ),
    )
