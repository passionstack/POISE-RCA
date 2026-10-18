"""Observed-only service topology for incomplete telemetry."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict

from poise_rca.domain import EvidenceRecord, Modality


class ObservedServiceNode(BaseModel):
    """A service seen in at least one available modality."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: str
    modalities: tuple[Modality, ...]
    evidence_ids: tuple[str, ...]
    connected: bool


class ObservedServiceEdge(BaseModel):
    """A call edge explicitly observed in trace evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    evidence_ids: tuple[str, ...]


class TopologyCompleteness(BaseModel):
    """Metadata preventing unobserved edges from becoming negative evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_evidence_count: int
    observed_edge_count: int
    trace_coverage_complete: bool = False
    missing_edges_mean: Literal["unknown"] = "unknown"


class ObservedServiceGraph(BaseModel):
    """Serializable view of the partial service graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[ObservedServiceNode, ...]
    edges: tuple[ObservedServiceEdge, ...]
    completeness: TopologyCompleteness

    def edge_status(self, source: str, target: str) -> Literal["observed", "unknown"]:
        if any(edge.source == source and edge.target == target for edge in self.edges):
            return "observed"
        return "unknown"


def build_observed_service_graph(
    evidence: Sequence[EvidenceRecord],
) -> ObservedServiceGraph:
    """Build a graph from explicit observations without completing absent edges."""

    graph: nx.DiGraph[str] = nx.DiGraph()
    node_modalities: dict[str, set[Modality]] = defaultdict(set)
    node_evidence: dict[str, set[str]] = defaultdict(set)
    edge_evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    trace_evidence_count = 0

    for record in evidence:
        graph.add_node(record.service)
        node_modalities[record.service].add(record.modality)
        node_evidence[record.service].add(record.evidence_id)
        if record.modality is not Modality.TRACES:
            continue
        trace_evidence_count += 1
        source = str(record.attributes.get("source_service", record.service)).strip()
        target_value = record.attributes.get("target_service")
        target = str(target_value).strip() if target_value is not None else ""
        if not source or not target:
            continue
        graph.add_edge(source, target)
        edge_evidence[(source, target)].add(record.evidence_id)
        for service in (source, target):
            node_modalities[service].add(Modality.TRACES)
            node_evidence[service].add(record.evidence_id)

    nodes = tuple(
        ObservedServiceNode(
            service=service,
            modalities=tuple(sorted(node_modalities[service], key=lambda item: item.value)),
            evidence_ids=tuple(sorted(node_evidence[service])),
            connected=graph.degree(service) > 0,
        )
        for service in sorted(graph.nodes)
    )
    edges = tuple(
        ObservedServiceEdge(
            source=source,
            target=target,
            evidence_ids=tuple(sorted(edge_evidence[(source, target)])),
        )
        for source, target in sorted(graph.edges)
    )
    return ObservedServiceGraph(
        nodes=nodes,
        edges=edges,
        completeness=TopologyCompleteness(
            trace_evidence_count=trace_evidence_count,
            observed_edge_count=len(edges),
        ),
    )
