from datetime import UTC, datetime, timedelta
from uuid import uuid4

from poise_rca.agents import (
    HypothesisBatch,
    HypothesisDraft,
    HypothesisPromptInput,
    InvestigationTaskDraft,
    SupervisorPlan,
    SupervisorPromptInput,
    WorkerAction,
    WorkerPromptInput,
)
from poise_rca.domain import (
    CandidateProposal,
    CaseContext,
    DiagnosisStatus,
    EvidencePolarity,
    EvidenceRecord,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    TaskKind,
)
from poise_rca.graph import DiagnosisState, build_diagnosis_graph
from poise_rca.graph.dependencies import GraphDependencies
from poise_rca.topology import build_observed_service_graph


class FakeHypothesisModel:
    def __init__(self, evidence_id: str) -> None:
        self.evidence_id = evidence_id

    def invoke(self, input: HypothesisPromptInput) -> object:
        assert input.findings == ()
        return HypothesisBatch(
            hypotheses=tuple(
                HypothesisDraft(
                    root_cause_service=service,
                    fault_type="delay",
                    causal_claim=f"{service} delay initiated the incident.",
                    origin_evidence_ids=(self.evidence_id,),
                    expected_observations=(f"{service} anomaly occurs first",),
                )
                for service in ("checkout", "payment", "frontend")
            )
        )


class AdaptiveSupervisorModel:
    def __init__(self) -> None:
        self.inputs: list[SupervisorPromptInput] = []

    def invoke(self, input: SupervisorPromptInput) -> object:
        self.inputs.append(input)
        concrete = {
            item.root_cause_service: item
            for item in input.hypotheses
            if item.root_cause_service != "OTHER/UNKNOWN"
        }
        shared = (input.evidence[0].evidence_id,)
        if not input.completed_task_signatures:
            return SupervisorPlan(
                tasks=(
                    self._comparison(
                        concrete["checkout"].hypothesis_id,
                        concrete["payment"].hypothesis_id,
                        "Compare checkout versus payment onset.",
                        shared,
                    ),
                    self._comparison(
                        concrete["payment"].hypothesis_id,
                        concrete["frontend"].hypothesis_id,
                        "Compare payment versus frontend onset.",
                        shared,
                    ),
                )
            ).model_dump()
        fraud = concrete.get("fraud")
        if fraud is not None and not any(
            fraud.hypothesis_id in signature for signature in input.completed_task_signatures
        ):
            return SupervisorPlan(
                tasks=(
                    self._comparison(
                        fraud.hypothesis_id,
                        concrete["checkout"].hypothesis_id,
                        "Compare fraud versus checkout onset.",
                        shared,
                    ),
                )
            ).model_dump()
        return SupervisorPlan(
            no_meaningful_competition=True,
            rationale="All currently useful comparisons have completed.",
        ).model_dump()

    @staticmethod
    def _comparison(
        left: str,
        right: str,
        question: str,
        evidence_ids: tuple[str, ...],
    ) -> InvestigationTaskDraft:
        return InvestigationTaskDraft(
            task_kind=TaskKind.COMPARISON,
            target_hypothesis_ids=(left, right),
            question=question,
            shared_evidence_ids=evidence_ids,
            discriminating_observations=("relative trace onset",),
            available_modalities=(Modality.TRACES,),
        )


class AdaptiveWorkerModel:
    def __init__(self) -> None:
        self.inputs: list[WorkerPromptInput] = []

    def invoke(self, input: WorkerPromptInput) -> object:
        self.inputs.append(input)
        if not input.query_history:
            start = datetime(2025, 1, 1, tzinfo=UTC)
            return WorkerAction(
                action="QUERY",
                tool_name="query_traces",
                arguments={
                    "service": input.hypotheses[0].root_cause_service,
                    "start_time": start.isoformat(),
                    "end_time": (start + timedelta(minutes=5)).isoformat(),
                },
            ).model_dump()
        newest_id = input.query_history[-1].evidence_ids[0]
        newest = next(item for item in input.evidence if item.evidence_id == newest_id)
        proposals: tuple[CandidateProposal, ...] = ()
        if newest.service == "fraud" and not any(
            item.root_cause_service == "fraud" for item in input.hypotheses
        ):
            proposals = (
                CandidateProposal(
                    service="fraud",
                    fault_type="delay",
                    causal_claim="fraud delay initiated the incident.",
                    evidence_ids=(newest.evidence_id,),
                    suggested_investigation="Compare fraud and checkout trace onset.",
                ),
            )
        return WorkerAction(
            action="REPORT",
            conclusion=(
                "The observed trace timing narrows this comparison but does not by itself "
                "exclude a contributing fault."
            ),
            cited_evidence_ids=(newest.evidence_id,),
            candidate_proposals=proposals,
        ).model_dump()


class AdaptiveToolExecutor:
    def __init__(self) -> None:
        self.tasks: list[InvestigationTask] = []

    def execute(self, task: InvestigationTask) -> InvestigationResult:
        self.tasks.append(task)
        service = "fraud" if "checkout versus payment" in task.question else task.service
        assert service is not None
        now = datetime.now(UTC)
        evidence = EvidenceRecord(
            evidence_id=str(uuid4()),
            modality=Modality.TRACES,
            service=service,
            observation=f"{service} trace anomaly observed during {task.question}",
            query_id=str(uuid4()),
            query_status=QueryStatus.SUCCESS_WITH_DATA,
            polarity=EvidencePolarity.NEUTRAL,
            source_reference=f"trace:{service}:integration",
            created_at=now,
        )
        return InvestigationResult(
            task_id=task.task_id,
            evidence=(evidence,),
            query_id=evidence.query_id,
            status=QueryStatus.SUCCESS_WITH_DATA,
            started_at=now,
            finished_at=now,
        )


def test_mas_graph_replans_after_worker_promotes_new_candidate() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    seed = EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.TRACES,
        service="checkout",
        observation="Checkout latency increased.",
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=EvidencePolarity.NEUTRAL,
        source_reference="trace:seed",
        created_at=start,
    )
    state = DiagnosisState(
        case=CaseContext(
            case_id="re2-ob-0000000000000001",
            incident_start=start,
            incident_end=start + timedelta(minutes=5),
            initial_symptom="Frontend latency increased.",
            entry_service="frontend",
            available_modalities=frozenset({Modality.TRACES}),
        ),
        service_inventory=frozenset({"checkout", "payment", "frontend", "fraud"}),
        topology=build_observed_service_graph((seed,)),
        evidence={seed.evidence_id: seed},
        hypotheses={},
        tasks={},
        results=(),
        conflicts=(),
    )
    supervisor = AdaptiveSupervisorModel()
    worker = AdaptiveWorkerModel()
    tools = AdaptiveToolExecutor()
    graph = build_diagnosis_graph(
        GraphDependencies(
            hypothesis_model=FakeHypothesisModel(seed.evidence_id),
            supervisor_model=supervisor,
            worker_model=worker,
            tool_registry=tools,
        )
    )

    output = graph.invoke(state.model_dump(mode="python"))

    assert output["supervisor_steps"] == 3
    assert len(output["tasks"]) == 3
    assert len(output["results"]) == 3
    assert "fraud" in {
        hypothesis.root_cause_service for hypothesis in output["hypotheses"].values()
    }
    fraud_id = next(
        hypothesis.hypothesis_id
        for hypothesis in output["hypotheses"].values()
        if hypothesis.root_cause_service == "fraud"
    )
    assert any(fraud_id in task.target_hypothesis_ids for task in output["tasks"].values())
    assert all(len(prompt.hypotheses) == 2 for prompt in worker.inputs)
    assert output["pending_candidate_proposals"] == ()
    assert output["report"].status is DiagnosisStatus.INDETERMINATE
