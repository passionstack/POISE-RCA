from datetime import UTC, datetime
from uuid import uuid4

from poise_rca.domain import EvidencePolarity, EvidenceRecord, Modality, QueryStatus
from poise_rca.topology import build_observed_service_graph


def _evidence(
    service: str,
    modality: Modality,
    *,
    attributes: dict[str, object] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=modality,
        service=service,
        observation="normalized observation",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference=f"{modality.value}.csv#query=1",
        attributes=attributes or {},
        created_at=datetime.now(UTC),
    )


def test_visible_trace_edges_are_retained() -> None:
    trace = _evidence(
        "frontend",
        Modality.TRACES,
        attributes={"source_service": "frontend", "target_service": "checkout"},
    )

    graph = build_observed_service_graph([trace])

    assert graph.edge_status("frontend", "checkout") == "observed"
    assert graph.edges[0].evidence_ids == (trace.evidence_id,)


def test_log_and_metric_only_services_become_unconnected_nodes() -> None:
    log = _evidence("email", Modality.LOGS)
    metric = _evidence("payment", Modality.METRICS)

    graph = build_observed_service_graph([log, metric])

    assert {node.service for node in graph.nodes} == {"email", "payment"}
    assert graph.edges == ()
    assert all(not node.connected for node in graph.nodes)


def test_absent_edges_remain_unknown_not_false() -> None:
    graph = build_observed_service_graph(
        [
            _evidence(
                "frontend",
                Modality.TRACES,
                attributes={"source_service": "frontend", "target_service": "checkout"},
            )
        ]
    )

    assert graph.edge_status("checkout", "payment") == "unknown"
    assert graph.completeness.missing_edges_mean == "unknown"
    assert graph.completeness.trace_coverage_complete is False
