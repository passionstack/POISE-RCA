import json
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

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
from poise_rca.cli import app
from poise_rca.data import DatasetManifest
from poise_rca.domain import (
    AnomalyFinding,
    InvestigationResult,
    InvestigationTask,
    Modality,
    QueryStatus,
    StaticAnalysisResult,
    TaskKind,
)


class FakeModel:
    def invoke(self, input: HypothesisPromptInput) -> object:
        service = next(
            (item.service for item in input.evidence if item.service != "OTHER/UNKNOWN"),
            "checkoutservice",
        )
        evidence_ids = tuple(item.evidence_id for item in input.evidence[:1])
        return HypothesisBatch(
            hypotheses=(
                HypothesisDraft(
                    root_cause_service=service,
                    fault_type="delay",
                    causal_claim=f"{service} delay may explain the incident",
                    origin_evidence_ids=evidence_ids,
                    missing_evidence=("independent trace propagation evidence",),
                ),
            )
        )


class InvalidModel:
    def invoke(self, input: HypothesisPromptInput) -> object:
        del input
        return {"not_hypotheses": "invalid"}


class TerminalSupervisorModel:
    def invoke(self, input: SupervisorPromptInput) -> object:
        del input
        return SupervisorPlan(
            no_meaningful_competition=True,
            rationale="CLI artifact test does not require telemetry investigation.",
        ).model_dump()


class WorkerMustNotRun:
    def invoke(self, input: WorkerPromptInput) -> object:
        raise AssertionError(f"worker unexpectedly ran: {input.task.task_id}")


class FailingStaticAnalyzer:
    def analyze(self, *args: object) -> object:
        del args
        raise RuntimeError("synthetic analyzer failure")


class CheckoutStaticAnalyzer:
    def analyze(self, case: object, bundle: object, inventory: frozenset[str]) -> object:
        del case, bundle
        service = "checkoutservice" if "checkoutservice" in inventory else sorted(inventory)[0]
        return StaticAnalysisResult(
            findings=(
                AnomalyFinding(
                    finding_id=str(uuid4()),
                    service=service,
                    modality=Modality.METRICS,
                    signal="latency",
                    observation=f"{service} latency deviated from baseline",
                    source_reference="synthetic/static-analysis",
                    analyzer="synthetic",
                    rank=1,
                ),
            ),
            statuses={
                Modality.METRICS: QueryStatus.SUCCESS_WITH_DATA,
                Modality.LOGS: QueryStatus.DATA_UNAVAILABLE,
                Modality.TRACES: QueryStatus.DATA_UNAVAILABLE,
            },
        )


class ToolFailureSupervisorModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, input: SupervisorPromptInput) -> object:
        self.calls += 1
        concrete = next(
            item for item in input.hypotheses if item.root_cause_service != "OTHER/UNKNOWN"
        )
        if self.calls == 1:
            return SupervisorPlan(
                tasks=(
                    InvestigationTaskDraft(
                        task_kind=TaskKind.CHALLENGE,
                        target_hypothesis_ids=(concrete.hypothesis_id,),
                        question="Can metrics distinguish this candidate despite tool failure?",
                        discriminating_observations=("incident-window deviation",),
                        available_modalities=(Modality.METRICS,),
                    ),
                )
            )
        return SupervisorPlan(
            no_meaningful_competition=True,
            rationale="The failed query leaves the candidate unresolved.",
        )


class ToolFailureWorkerModel:
    def invoke(self, input: WorkerPromptInput) -> object:
        if not input.query_history:
            return WorkerAction(
                action="QUERY",
                tool_name="query_metrics",
                arguments={"service": input.hypotheses[0].root_cause_service},
            )
        return WorkerAction(
            action="REPORT",
            conclusion="The metric query failed, so this candidate cannot be distinguished.",
            cited_evidence_ids=(input.evidence[0].evidence_id,),
            unresolved_questions=("Retry the metric query after restoring the data source.",),
        )


class FailingToolRegistry:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def execute(self, task: InvestigationTask) -> InvestigationResult:
        now = datetime.now(UTC)
        return InvestigationResult(
            task_id=task.task_id,
            query_id=str(uuid4()),
            status=QueryStatus.TOOL_ERROR,
            error="synthetic tool failure",
            started_at=now,
            finished_at=now,
        )


def _patch_mas_models(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "poise_rca.cli._create_supervisor_model",
        lambda settings: TerminalSupervisorModel(),
    )
    monkeypatch.setattr(
        "poise_rca.cli._create_worker_model",
        lambda settings: WorkerMustNotRun(),
    )


class FakeTraceRuntime:
    callbacks: tuple[object, ...] = ()

    def __init__(self, *, fail_flush: bool = False) -> None:
        self.fail_flush = fail_flush
        self.flushed = False
        self.outputs: list[object] = []
        self.events: list[tuple[str, dict[str, object], str]] = []

    @property
    def trace_id(self) -> str | None:
        return "trace-test"

    def event(
        self,
        name: str,
        attributes: dict[str, object],
        observation_type: str = "chain",
    ) -> None:
        self.events.append((name, attributes, observation_type))

    @contextmanager
    def observation(
        self,
        name: str,
        observation_type: str,
        *,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[object | None]:
        del name, observation_type, input, metadata
        yield None

    def set_trace_output(self, output: object) -> None:
        self.outputs.append(output)

    def flush(self) -> None:
        self.flushed = True
        if self.fail_flush:
            raise RuntimeError("trace collector unavailable")


def _dataset(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "fixtures" / "synthetic_case"
    case = tmp_path / "dataset" / "re2_ob_checkoutservice_delay_1"
    shutil.copytree(source, case)
    return case.parent


def _sample_config(tmp_path: Path, manifest: Path) -> Path:
    path = tmp_path / "sampling.yaml"
    path.write_text(
        "\n".join(
            (
                "dataset:",
                "  version: RCAEval-RE2-OB",
                "selection:",
                "  seed: 17",
                "  size: 1",
                "  constraints: {}",
                f"manifest: {manifest.as_posix()}",
            )
        ),
        encoding="utf-8",
    )
    return path


def _runtime_config(tmp_path: Path, dataset_root: Path, results_root: Path) -> Path:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "\n".join(
            (
                "dataset:",
                f"  root: {dataset_root.as_posix()}",
                "results:",
                f"  root: {results_root.as_posix()}",
                "model:",
                "  name: fake-model",
                "mas:",
                "  max_supervisor_steps: 4",
                "tools:",
                "  per_tool_timeout_seconds: 1",
                "tracing:",
                "  enabled: false",
                "  export_raw_telemetry: false",
            )
        ),
        encoding="utf-8",
    )
    return path


def _create_manifest(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: object,
) -> tuple[Path, Path]:
    dataset_root = _dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    config = _sample_config(tmp_path, manifest_path)
    monkeypatch.setenv("POISE_RCA_DATASET_ROOT", str(dataset_root))
    result = runner.invoke(app, ["sample", "--config", str(config)])
    assert result.exit_code == 0, result.output
    return dataset_root, manifest_path


def test_sample_diagnose_and_batch_write_schema_valid_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    dataset_root, manifest_path = _create_manifest(runner, tmp_path, monkeypatch)
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    case_id = manifest.cases[0].safe_case_id
    results_root = tmp_path / "results"
    runtime_config = _runtime_config(tmp_path, dataset_root, results_root)
    traces: list[FakeTraceRuntime] = []
    monkeypatch.setattr("poise_rca.cli._create_hypothesis_model", lambda settings: FakeModel())
    _patch_mas_models(monkeypatch)
    monkeypatch.setattr(
        "poise_rca.cli.create_trace_runtime",
        lambda settings: traces.append(FakeTraceRuntime()) or traces[-1],
    )

    diagnose = runner.invoke(
        app,
        [
            "diagnose",
            "--case-id",
            case_id,
            "--manifest",
            str(manifest_path),
            "--config",
            str(runtime_config),
        ],
    )
    assert diagnose.exit_code == 0, diagnose.output
    reports = list(results_root.rglob("report.json"))
    states = list(results_root.rglob("state.json"))
    assert len(reports) == len(states) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["case_id"] == case_id
    assert report["trace_id"] == "trace-test"
    assert report["run_id"] in str(reports[0])
    assert report["status"] in {
        "CONFIRMED",
        "NARROWED",
        "INDETERMINATE",
        "OBSERVATION_FAILURE",
    }
    event_payloads = {name: attributes for name, attributes, _ in traces[-1].events}
    assert event_payloads["hypotheses.generated"]["hypothesis_ids"]
    assert event_payloads["supervisor.planned"]["step"] == 1
    assert event_payloads["report.generated"]["supervisor_steps"] == 1
    assert "hypothesis_states" in event_payloads["report.generated"]
    serialized_events = json.dumps(event_payloads)
    assert all(
        raw_key not in serialized_events
        for raw_key in ("raw_logs", "raw_metrics", "raw_traces", "telemetry")
    )
    assert traces[-1].flushed

    batch = runner.invoke(
        app,
        [
            "batch",
            "--manifest",
            str(manifest_path),
            "--config",
            str(runtime_config),
        ],
    )
    assert batch.exit_code == 0, batch.output
    summaries = list(results_root.rglob("summary.json"))
    assert summaries
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert traces[-1].flushed


def test_invalid_model_output_and_tracing_failure_degrade_safely(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    dataset_root, manifest_path = _create_manifest(runner, tmp_path, monkeypatch)
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    results_root = tmp_path / "results"
    runtime_config = _runtime_config(tmp_path, dataset_root, results_root)
    trace = FakeTraceRuntime(fail_flush=True)
    monkeypatch.setattr("poise_rca.cli._create_hypothesis_model", lambda settings: InvalidModel())
    _patch_mas_models(monkeypatch)
    monkeypatch.setattr("poise_rca.cli.create_trace_runtime", lambda settings: trace)

    result = runner.invoke(
        app,
        [
            "diagnose",
            "--case-id",
            manifest.cases[0].safe_case_id,
            "--manifest",
            str(manifest_path),
            "--config",
            str(runtime_config),
        ],
    )

    assert result.exit_code == 0, result.output
    report_path = next(results_root.rglob("report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] in {
        "INDETERMINATE",
        "OBSERVATION_FAILURE",
    }
    assert report["root_cause"] is None
    assert trace.flushed


def test_static_analyzer_failure_becomes_observation_failure_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    dataset_root, manifest_path = _create_manifest(runner, tmp_path, monkeypatch)
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    results_root = tmp_path / "results"
    runtime_config = _runtime_config(tmp_path, dataset_root, results_root)
    monkeypatch.setattr("poise_rca.cli.RCAEvalMMBaroAnalyzer", lambda: FailingStaticAnalyzer())
    monkeypatch.setattr("poise_rca.cli._create_hypothesis_model", lambda settings: FakeModel())
    _patch_mas_models(monkeypatch)

    result = runner.invoke(
        app,
        [
            "diagnose",
            "--case-id",
            manifest.cases[0].safe_case_id,
            "--manifest",
            str(manifest_path),
            "--config",
            str(runtime_config),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(next(results_root.rglob("report.json")).read_text(encoding="utf-8"))
    state = json.loads(next(results_root.rglob("state.json")).read_text(encoding="utf-8"))
    assert report["status"] == "OBSERVATION_FAILURE"
    assert report["root_cause"] is None
    assert all(
        status in {"TOOL_ERROR", "DATA_UNAVAILABLE"}
        for signature, status in state["query_statuses"].items()
        if signature.startswith("static|")
    )


def test_worker_tool_failure_preserves_uncertainty_and_cli_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    dataset_root, manifest_path = _create_manifest(runner, tmp_path, monkeypatch)
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    results_root = tmp_path / "results"
    runtime_config = _runtime_config(tmp_path, dataset_root, results_root)
    supervisor = ToolFailureSupervisorModel()
    monkeypatch.setattr("poise_rca.cli._create_hypothesis_model", lambda settings: FakeModel())
    monkeypatch.setattr("poise_rca.cli._create_supervisor_model", lambda settings: supervisor)
    monkeypatch.setattr(
        "poise_rca.cli._create_worker_model", lambda settings: ToolFailureWorkerModel()
    )
    monkeypatch.setattr("poise_rca.cli.ToolRegistry", FailingToolRegistry)
    monkeypatch.setattr("poise_rca.cli.RCAEvalMMBaroAnalyzer", lambda: CheckoutStaticAnalyzer())

    result = runner.invoke(
        app,
        [
            "diagnose",
            "--case-id",
            manifest.cases[0].safe_case_id,
            "--manifest",
            str(manifest_path),
            "--config",
            str(runtime_config),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(next(results_root.rglob("report.json")).read_text(encoding="utf-8"))
    assert report["status"] == "INDETERMINATE"
    assert report["root_cause"] is None
    assert report["worker_conclusions"][0]["conclusion"].startswith("The metric query failed")
    assert report["unresolved_questions"] == [
        "Retry the metric query after restoring the data source."
    ]
    assert any("TOOL_ERROR" in item for item in report["missing_or_unavailable_evidence"])
