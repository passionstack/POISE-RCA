from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from poise_rca.agents import (
    HypothesisBatch,
    HypothesisDraft,
    HypothesisPromptInput,
    HypothesisTransitionRecommendation,
    InvestigationTaskDraft,
    SupervisorPlan,
    SupervisorPromptInput,
    WorkerAction,
    WorkerPromptInput,
)
from poise_rca.domain import (
    AnomalyFinding,
    CandidateProposal,
    CaseContext,
    CausalRole,
    DiagnosisStatus,
    EvidencePolarity,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    TaskKind,
)
from poise_rca.graph import DiagnosisState, build_diagnosis_graph
from poise_rca.graph.dependencies import GraphDependencies
from poise_rca.reporting import DiagnosisReport
from poise_rca.topology import build_observed_service_graph


class ModelMustNotRun:
    def invoke(self, input: object) -> object:
        del input
        raise AssertionError("preloaded hypotheses must bypass generation")


class StaticFindingHypothesisModel:
    def __init__(self, roles: dict[str, CausalRole] | None = None) -> None:
        self.roles = roles or {}
        self.inputs: list[HypothesisPromptInput] = []

    def invoke(self, input: HypothesisPromptInput) -> object:
        self.inputs.append(input)
        evidence_by_service = {item.service: item.evidence_id for item in input.evidence}
        return HypothesisBatch(
            hypotheses=tuple(
                HypothesisDraft(
                    root_cause_service=finding.service,
                    fault_type="delay",
                    causal_claim=f"{finding.service} anomaly may explain the incident",
                    causal_role=self.roles.get(finding.service, CausalRole.UNKNOWN),
                    origin_evidence_ids=(evidence_by_service[finding.service],),
                    expected_observations=(f"causal propagation from {finding.service}",),
                )
                for finding in input.findings
            )
        )


class TerminalSupervisor:
    def invoke(self, input: SupervisorPromptInput) -> object:
        del input
        return SupervisorPlan(
            no_meaningful_competition=True,
            rationale="No additional discriminating task is justified.",
        )


class EmptyReplanningSupervisor:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, input: SupervisorPromptInput) -> object:
        del input
        self.calls += 1
        return SupervisorPlan(
            no_meaningful_competition=False,
            rationale="Keep reconsidering the same unresolved candidates.",
        )


class TwoStepSupervisor:
    def __init__(
        self,
        first_plan: Callable[[SupervisorPromptInput], SupervisorPlan],
        second_plan: Callable[[SupervisorPromptInput], SupervisorPlan] | None = None,
    ) -> None:
        self.first_plan = first_plan
        self.second_plan = second_plan
        self.inputs: list[SupervisorPromptInput] = []

    def invoke(self, input: SupervisorPromptInput) -> object:
        self.inputs.append(input)
        if len(self.inputs) == 1:
            return self.first_plan(input)
        if self.second_plan is not None:
            return self.second_plan(input)
        return SupervisorPlan(
            no_meaningful_competition=True,
            rationale="The completed task is sufficient for this synthetic scenario.",
        )


class QueryThenReportWorker:
    def __init__(
        self,
        *,
        conclusion: str,
        proposal_service: str | None = None,
        cite_initial_when_query_empty: bool = False,
    ) -> None:
        self.conclusion = conclusion
        self.proposal_service = proposal_service
        self.cite_initial_when_query_empty = cite_initial_when_query_empty
        self.inputs: list[WorkerPromptInput] = []

    def invoke(self, input: WorkerPromptInput) -> object:
        self.inputs.append(input)
        if not input.query_history:
            return WorkerAction(
                action="QUERY",
                tool_name="query_metrics",
                arguments={"service": input.hypotheses[0].root_cause_service},
            )
        cited_ids = input.query_history[-1].evidence_ids
        if not cited_ids and self.cite_initial_when_query_empty:
            cited_ids = tuple(item.evidence_id for item in input.evidence[:1])
        proposals: tuple[CandidateProposal, ...] = ()
        if self.proposal_service is not None:
            proposals = (
                CandidateProposal(
                    service=self.proposal_service,
                    fault_type="delay",
                    causal_claim=f"{self.proposal_service} delay may explain the incident",
                    evidence_ids=cited_ids,
                    suggested_investigation=(
                        f"Compare {self.proposal_service} with current candidates."
                    ),
                ),
            )
        unresolved = (
            ("The unavailable modality still needs collection.",)
            if not input.query_history[-1].evidence_ids
            else ()
        )
        return WorkerAction(
            action="REPORT",
            conclusion=self.conclusion,
            cited_evidence_ids=cited_ids,
            candidate_proposals=proposals,
            unresolved_questions=unresolved,
        )


class SyntheticRegistry:
    def __init__(
        self,
        records: Callable[[InvestigationTask], tuple[EvidenceRecord, ...]],
        status: QueryStatus = QueryStatus.SUCCESS_WITH_DATA,
    ) -> None:
        self.records = records
        self.status = status
        self.tasks: list[InvestigationTask] = []

    def execute(self, task: InvestigationTask) -> InvestigationResult:
        self.tasks.append(task)
        evidence = self.records(task)
        return InvestigationResult(
            task_id=task.task_id,
            evidence=evidence,
            query_id=str(uuid4()),
            status=self.status,
            error=None if self.status is QueryStatus.SUCCESS_WITH_DATA else self.status.value,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )


def _case() -> CaseContext:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return CaseContext(
        case_id="re2-ob-0000000000000001",
        incident_start=start,
        incident_end=start + timedelta(minutes=5),
        initial_symptom="Frontend latency increased.",
        entry_service="frontend",
        available_modalities=frozenset({Modality.METRICS}),
    )


def _evidence(
    service: str,
    observation: str,
    *,
    polarity: EvidencePolarity = EvidencePolarity.NEUTRAL,
    attributes: dict[str, object] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(uuid4()),
        modality=Modality.METRICS,
        service=service,
        observation=observation,
        query_id=str(uuid4()),
        query_status=QueryStatus.SUCCESS_WITH_DATA,
        polarity=polarity,
        source_reference=f"synthetic/metrics.csv#service={service}",
        attributes=attributes or {},
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _finding(service: str, rank: int) -> AnomalyFinding:
    return AnomalyFinding(
        finding_id=str(uuid4()),
        service=service,
        modality=Modality.METRICS,
        signal="latency",
        observation=f"{service} latency deviated from baseline",
        source_reference=f"synthetic/metrics.csv#service={service}",
        analyzer="synthetic-static-analysis",
        score=1.0 / rank,
        rank=rank,
    )


def _candidate(
    service: str,
    evidence: EvidenceRecord,
    *,
    role: CausalRole = CausalRole.UNKNOWN,
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=str(uuid4()),
        root_cause_service=service,
        fault_type="delay",
        causal_claim=f"{service} delay may explain frontend latency",
        causal_role=role,
        origin_evidence_ids=(evidence.evidence_id,),
        expected_observations=(f"causal propagation from {service}",),
        status=HypothesisStatus.ACTIVE,
    )


def _state(
    hypotheses: tuple[Hypothesis, ...] = (),
    *,
    evidence: tuple[EvidenceRecord, ...] = (),
    findings: tuple[AnomalyFinding, ...] = (),
    inventory: frozenset[str] = frozenset({"frontend", "checkout", "payment", "inventory"}),
    max_supervisor_steps: int = 4,
) -> DiagnosisState:
    return DiagnosisState(
        case=_case(),
        service_inventory=inventory,
        topology=build_observed_service_graph(evidence),
        initial_findings={item.finding_id: item for item in findings},
        evidence={item.evidence_id: item for item in evidence},
        hypotheses={item.hypothesis_id: item for item in hypotheses},
        tasks={},
        results=(),
        query_statuses={"static|metrics": QueryStatus.SUCCESS_WITH_DATA},
        conflicts=(),
        max_supervisor_steps=max_supervisor_steps,
    )


def _task(
    input: SupervisorPromptInput,
    *,
    kind: TaskKind,
    targets: tuple[str, ...],
    question: str,
) -> SupervisorPlan:
    return SupervisorPlan(
        tasks=(
            InvestigationTaskDraft(
                task_kind=kind,
                target_hypothesis_ids=targets,
                question=question,
                shared_evidence_ids=tuple(
                    identifier
                    for item in input.hypotheses
                    if item.hypothesis_id in targets
                    for identifier in item.supporting_evidence_ids
                ),
                discriminating_observations=("incident-window causal ordering",),
                available_modalities=(Modality.METRICS,),
            ),
        ),
    )


def _run(
    state: DiagnosisState,
    *,
    hypothesis_model: object,
    supervisor_model: object,
    worker_model: object | None = None,
    registry: SyntheticRegistry | None = None,
) -> tuple[DiagnosisState, DiagnosisReport]:
    graph = build_diagnosis_graph(
        GraphDependencies(
            hypothesis_model=hypothesis_model,  # type: ignore[arg-type]
            supervisor_model=supervisor_model,  # type: ignore[arg-type]
            worker_model=worker_model,  # type: ignore[arg-type]
            tool_registry=registry or SyntheticRegistry(lambda task: ()),
            max_supervisor_steps=state.max_supervisor_steps,
        )
    )
    output = graph.invoke(state.model_dump(mode="python"))
    state_values = {
        key: value for key, value in output.items() if key in DiagnosisState.model_fields
    }
    return DiagnosisState.model_validate(state_values), DiagnosisReport.model_validate(
        output["report"]
    )


def test_static_findings_seed_more_than_two_initial_hypotheses() -> None:
    services = ("checkout", "payment", "inventory")
    findings = tuple(_finding(service, rank) for rank, service in enumerate(services, start=1))
    evidence = tuple(_evidence(service, f"{service} static anomaly") for service in services)
    model = StaticFindingHypothesisModel()

    final, report = _run(
        _state(evidence=evidence, findings=findings),
        hypothesis_model=model,
        supervisor_model=TerminalSupervisor(),
    )

    concrete = [
        item for item in final.hypotheses.values() if item.root_cause_service != "OTHER/UNKNOWN"
    ]
    assert {item.root_cause_service for item in concrete} == set(services)
    assert len(concrete) == 3
    assert all(item.status is HypothesisStatus.ACTIVE for item in concrete)
    assert len(model.inputs[0].findings) == 3
    assert report.status is DiagnosisStatus.INDETERMINATE


def test_comparative_worker_returns_discriminating_evidence_and_grounded_transitions() -> None:
    checkout_seed = _evidence("checkout", "checkout static anomaly")
    payment_seed = _evidence("payment", "payment static anomaly")
    checkout = _candidate("checkout", checkout_seed)
    payment = _candidate("payment", payment_seed)

    def first(input: SupervisorPromptInput) -> SupervisorPlan:
        return _task(
            input,
            kind=TaskKind.COMPARISON,
            targets=(checkout.hypothesis_id, payment.hypothesis_id),
            question="Which candidate is the initiating cause: checkout or payment?",
        )

    def second(input: SupervisorPromptInput) -> SupervisorPlan:
        support = next(
            item.evidence_id for item in input.evidence if "precedes" in item.observation
        )
        contradiction = next(
            item.evidence_id for item in input.evidence if "remains normal" in item.observation
        )
        return SupervisorPlan(
            transition_recommendations=(
                HypothesisTransitionRecommendation(
                    hypothesis_id=checkout.hypothesis_id,
                    target_status=HypothesisStatus.LEADING,
                    rationale="Checkout change precedes downstream symptoms.",
                    evidence_ids=(support,),
                ),
                HypothesisTransitionRecommendation(
                    hypothesis_id=payment.hypothesis_id,
                    target_status=HypothesisStatus.REFUTED,
                    rationale="Payment lacks its necessary predicted anomaly.",
                    evidence_ids=(contradiction,),
                ),
            ),
            no_meaningful_competition=True,
        )

    registry = SyntheticRegistry(
        lambda task: (
            _evidence(
                "checkout",
                "checkout anomaly precedes frontend degradation",
                polarity=EvidencePolarity.SUPPORT,
                attributes={"causal_support": True, "comparative": True},
            ),
            _evidence(
                "payment",
                "payment latency remains normal during the incident",
                polarity=EvidencePolarity.CONTRADICT,
                attributes={"necessary_prediction": True, "adequate_coverage": True},
            ),
        )
    )
    final, report = _run(
        _state((checkout, payment), evidence=(checkout_seed, payment_seed)),
        hypothesis_model=ModelMustNotRun(),
        supervisor_model=TwoStepSupervisor(first, second),
        worker_model=QueryThenReportWorker(
            conclusion="The observed ordering favors checkout over payment."
        ),
        registry=registry,
    )

    assert final.hypotheses[checkout.hypothesis_id].status is HypothesisStatus.LEADING
    assert final.hypotheses[payment.hypothesis_id].status is HypothesisStatus.REFUTED
    assert report.status is DiagnosisStatus.CONFIRMED
    assert report.worker_conclusions[0].conclusion == (
        "The observed ordering favors checkout over payment."
    )
    assert len(report.worker_conclusions[0].cited_evidence_ids) == 2


def test_worker_can_promote_an_initially_omitted_service_candidate() -> None:
    checkout_seed = _evidence("checkout", "checkout static anomaly")
    checkout = _candidate("checkout", checkout_seed)

    def first(input: SupervisorPromptInput) -> SupervisorPlan:
        return _task(
            input,
            kind=TaskKind.CHALLENGE,
            targets=(checkout.hypothesis_id,),
            question="Can another observed service better explain the checkout anomaly?",
        )

    registry = SyntheticRegistry(
        lambda task: (
            _evidence(
                "payment",
                "payment degradation precedes checkout latency",
                polarity=EvidencePolarity.SUPPORT,
                attributes={"source_service": "payment", "target_service": "checkout"},
            ),
        )
    )
    final, report = _run(
        _state((checkout,), evidence=(checkout_seed,)),
        hypothesis_model=ModelMustNotRun(),
        supervisor_model=TwoStepSupervisor(first),
        worker_model=QueryThenReportWorker(
            conclusion="Payment is a credible omitted alternative.",
            proposal_service="payment",
        ),
        registry=registry,
    )

    promoted = [item for item in final.hypotheses.values() if item.root_cause_service == "payment"]
    assert len(promoted) == 1
    assert promoted[0].status is HypothesisStatus.ACTIVE
    assert report.candidate_promotions == ("payment",)
    assert report.root_cause is None


def test_unavailable_telemetry_defers_but_does_not_refute_candidate() -> None:
    checkout_seed = _evidence("checkout", "checkout static anomaly")
    checkout = _candidate("checkout", checkout_seed)

    def first(input: SupervisorPromptInput) -> SupervisorPlan:
        return _task(
            input,
            kind=TaskKind.CHALLENGE,
            targets=(checkout.hypothesis_id,),
            question="Does metric evidence distinguish checkout from unobserved causes?",
        )

    def second(input: SupervisorPromptInput) -> SupervisorPlan:
        del input
        return SupervisorPlan(
            transition_recommendations=(
                HypothesisTransitionRecommendation(
                    hypothesis_id=checkout.hypothesis_id,
                    target_status=HypothesisStatus.DEFERRED,
                    rationale="Required metric detail is unavailable.",
                    evidence_ids=(checkout_seed.evidence_id,),
                ),
            ),
            no_meaningful_competition=True,
        )

    final, report = _run(
        _state((checkout,), evidence=(checkout_seed,)),
        hypothesis_model=ModelMustNotRun(),
        supervisor_model=TwoStepSupervisor(first, second),
        worker_model=QueryThenReportWorker(
            conclusion="The available data cannot distinguish checkout from alternatives.",
            cite_initial_when_query_empty=True,
        ),
        registry=SyntheticRegistry(lambda task: (), QueryStatus.DATA_UNAVAILABLE),
    )

    assert final.hypotheses[checkout.hypothesis_id].status is HypothesisStatus.DEFERRED
    assert report.root_cause is None
    assert report.unresolved_questions == ("The unavailable modality still needs collection.",)
    assert any("DATA_UNAVAILABLE" in item for item in report.missing_or_unavailable_evidence)


def test_nonexclusive_causal_roles_remain_active() -> None:
    services = ("inventory", "checkout")
    findings = tuple(_finding(service, rank) for rank, service in enumerate(services, start=1))
    evidence = tuple(_evidence(service, f"{service} static anomaly") for service in services)
    roles = {
        "inventory": CausalRole.INITIATING_CAUSE,
        "checkout": CausalRole.PROPAGATED_SYMPTOM,
    }

    final, report = _run(
        _state(evidence=evidence, findings=findings),
        hypothesis_model=StaticFindingHypothesisModel(roles),
        supervisor_model=TerminalSupervisor(),
    )

    concrete = [
        item for item in final.hypotheses.values() if item.root_cause_service != "OTHER/UNKNOWN"
    ]
    assert {item.causal_role for item in concrete} == set(roles.values())
    assert all(item.status is HypothesisStatus.ACTIVE for item in concrete)
    assert report.status is DiagnosisStatus.NARROWED
    assert report.root_cause is None


def test_challenge_finding_no_alternative_does_not_promote_a_leader() -> None:
    checkout_seed = _evidence("checkout", "checkout static anomaly")
    checkout = _candidate("checkout", checkout_seed)

    def first(input: SupervisorPromptInput) -> SupervisorPlan:
        return _task(
            input,
            kind=TaskKind.CHALLENGE,
            targets=(checkout.hypothesis_id,),
            question="Can the current evidence reveal an alternative explanation?",
        )

    final, report = _run(
        _state((checkout,), evidence=(checkout_seed,)),
        hypothesis_model=ModelMustNotRun(),
        supervisor_model=TwoStepSupervisor(first),
        worker_model=QueryThenReportWorker(
            conclusion="This query found no alternative explanation in the observed metrics."
        ),
        registry=SyntheticRegistry(
            lambda task: (_evidence("checkout", "no additional anomalous service was observed"),)
        ),
    )

    assert final.hypotheses[checkout.hypothesis_id].status is HypothesisStatus.ACTIVE
    assert report.root_cause is None
    assert report.status is not DiagnosisStatus.CONFIRMED


def test_repeated_supervisor_plans_hit_step_guard_without_forced_attribution() -> None:
    checkout_seed = _evidence("checkout", "checkout static anomaly")
    checkout = _candidate("checkout", checkout_seed)
    supervisor = EmptyReplanningSupervisor()

    final, report = _run(
        _state((checkout,), evidence=(checkout_seed,), max_supervisor_steps=3),
        hypothesis_model=ModelMustNotRun(),
        supervisor_model=supervisor,
    )

    assert supervisor.calls == 3
    assert final.supervisor_steps == 3
    assert report.supervisor_steps == 3
    assert report.status is DiagnosisStatus.INDETERMINATE
    assert report.stop_reason == "MAX_SUPERVISOR_STEPS"
    assert report.root_cause is None
    assert report.remaining_candidates[0].hypothesis_id == checkout.hypothesis_id
