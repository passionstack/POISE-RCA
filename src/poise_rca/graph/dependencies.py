"""Dependency injection contracts for deterministic graph execution."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from poise_rca.agents.evidence_auditor import AuditModel
from poise_rca.agents.hypothesis_generator import HypothesisModel
from poise_rca.agents.supervisor import SupervisorModel
from poise_rca.agents.worker import WorkerModel, WorkerToolDescription
from poise_rca.domain import InvestigationResult, InvestigationTask
from poise_rca.observability import ObservationType
from poise_rca.reporting import RunMetadata


class ToolExecutor(Protocol):
    def execute(self, task: InvestigationTask) -> InvestigationResult:
        """Execute one bounded worker-authored telemetry query."""


class GraphTracer(Protocol):
    def event(
        self,
        name: str,
        attributes: dict[str, object],
        observation_type: ObservationType = "chain",
    ) -> None:
        """Record a fail-open graph event."""


class NullGraphTracer:
    def event(
        self,
        name: str,
        attributes: dict[str, object],
        observation_type: ObservationType = "chain",
    ) -> None:
        del name, attributes, observation_type


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    hypothesis_model: HypothesisModel
    tool_registry: ToolExecutor
    supervisor_model: SupervisorModel | None = None
    worker_model: WorkerModel | None = None
    worker_tools: tuple[WorkerToolDescription, ...] = field(
        default_factory=lambda: tuple(
            WorkerToolDescription(
                name=f"query_{modality}",
                description=f"Query normalized {modality} telemetry for a service and time range.",
                argument_schema={
                    "type": "object",
                    "required": ["service", "start_time", "end_time"],
                },
            )
            for modality in ("logs", "metrics", "traces")
        )
    )
    max_supervisor_steps: int = 10
    tracer: GraphTracer = NullGraphTracer()
    audit_model: AuditModel | None = None
    run_metadata: RunMetadata = field(
        default_factory=lambda: RunMetadata(
            run_id=str(uuid4()),
            code_sha="unknown",
            config_fingerprint="unknown",
            latency_ms=0,
        )
    )

    @staticmethod
    def clock() -> datetime:
        return datetime.now(UTC)
