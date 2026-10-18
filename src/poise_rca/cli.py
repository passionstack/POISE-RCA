"""Operational command-line workflows for local POISE-RCA experiments."""

import json
import os
import subprocess
import warnings
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, cast
from uuid import uuid4

import typer
import yaml  # type: ignore[import-untyped]
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, SecretStr

from poise_rca.agents import (
    HypothesisModel,
    ModelSettings,
    SupervisorModel,
    WorkerModel,
    create_hypothesis_model,
    create_supervisor_model,
    create_worker_model,
)
from poise_rca.analysis import RCAEvalMMBaroAnalyzer, StaticAnalyzer, finding_to_evidence
from poise_rca.data import (
    CaseRecord,
    DatasetManifest,
    SamplingConfig,
    SamplingConstraints,
    TelemetryBundle,
    create_stratified_manifest,
    discover_re2_ob,
    load_re2_ob_case,
    to_agent_case_context,
    write_manifest,
)
from poise_rca.domain import (
    AnomalyFinding,
    Modality,
    QueryStatus,
    StaticAnalysisResult,
)
from poise_rca.graph import DiagnosisState, build_diagnosis_graph
from poise_rca.graph.dependencies import GraphDependencies
from poise_rca.graph.nodes import WorkflowState
from poise_rca.observability import LangFuseSettings, TraceRuntime, create_trace_runtime
from poise_rca.reporting import DiagnosisReport, RunMetadata, write_report
from poise_rca.tools import ToolRegistry
from poise_rca.topology import build_observed_service_graph

app = typer.Typer(
    name="poise-rca",
    help="Evidence-grounded root-cause diagnosis under partial observability.",
    no_args_is_help=True,
)


class DatasetSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    root: Path = Path("data/raw/rcaeval")


class ResultsSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    root: Path = Path("results")


class CliModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = "gpt-4.1-mini"
    temperature: float = Field(default=0.0, ge=0, le=2)


class MASSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    max_supervisor_steps: PositiveInt = 10


class ToolSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    per_tool_timeout_seconds: PositiveInt = 30


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    dataset: DatasetSettings = DatasetSettings()
    results: ResultsSettings = ResultsSettings()
    model: CliModelSettings = CliModelSettings()
    mas: MASSettings = MASSettings()
    tools: ToolSettings = ToolSettings()
    tracing: LangFuseSettings = LangFuseSettings()


def _yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def _load_runtime_settings(path: Path) -> RuntimeSettings:
    raw = _yaml_mapping(path)
    dataset = dict(raw.get("dataset") or {})
    results = dict(raw.get("results") or {})
    model = dict(raw.get("model") or {})
    tracing = dict(raw.get("tracing") or {})
    if value := os.getenv("POISE_RCA_DATASET_ROOT"):
        dataset["root"] = value
    if value := os.getenv("POISE_RCA_RESULTS_ROOT"):
        results["root"] = value
    if value := os.getenv("POISE_RCA_MODEL"):
        model["name"] = value
    tracing.update(
        {
            key: value
            for key, value in {
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
                "base_url": os.getenv("LANGFUSE_BASE_URL"),
                "environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT"),
            }.items()
            if value
        }
    )
    return RuntimeSettings.model_validate(
        {
            **raw,
            "dataset": dataset,
            "results": results,
            "model": model,
            "tracing": tracing,
        }
    )


def _model_settings(settings: RuntimeSettings) -> ModelSettings:
    api_key = os.getenv("OPENAI_API_KEY")
    return ModelSettings(
        model=settings.model.name,
        api_key=SecretStr(api_key) if api_key else None,
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=settings.model.temperature,
    )


def _create_hypothesis_model(settings: RuntimeSettings) -> HypothesisModel:
    return create_hypothesis_model(_model_settings(settings))


def _create_supervisor_model(settings: RuntimeSettings) -> SupervisorModel:
    return create_supervisor_model(_model_settings(settings))


def _create_worker_model(settings: RuntimeSettings) -> WorkerModel:
    return create_worker_model(_model_settings(settings))


def _services_by_modality(bundle: TelemetryBundle) -> dict[Modality, tuple[str, ...]]:
    frames = {
        Modality.METRICS: bundle.metrics,
        Modality.LOGS: bundle.logs,
        Modality.TRACES: bundle.traces,
    }
    services_by_modality: dict[Modality, tuple[str, ...]] = {}
    for modality, frame in frames.items():
        if frame is None:
            services_by_modality[modality] = ()
            continue
        service_column = next(
            (
                column
                for column in (
                    "service",
                    "service_name",
                    "servicename",
                    "container_name",
                    "containername",
                    "source_service",
                    "caller",
                )
                if column in frame.columns
            ),
            None,
        )
        if service_column is not None:
            services = {str(value) for value in frame[service_column].dropna() if str(value)}
        elif modality is Modality.METRICS:
            ignored = {"timestamp", "time", "ts", "metric", "metric_name", "value"}
            services = {
                str(column).split("_", 1)[0]
                for column in frame.columns
                if str(column) not in ignored and "_" in str(column)
            }
        else:
            services = set()
        services_by_modality[modality] = tuple(sorted(services))
    return services_by_modality


def _service_inventory(bundle: TelemetryBundle) -> frozenset[str]:
    services = {
        service
        for modality_services in _services_by_modality(bundle).values()
        for service in modality_services
    }
    if bundle.entry_service:
        services.add(bundle.entry_service)
    return frozenset(services)


def _initialize_diagnosis_state(
    bundle: TelemetryBundle,
    analyzer: StaticAnalyzer,
    max_supervisor_steps: int = 10,
) -> DiagnosisState:
    """Run static analysis before hypothesis generation and build safe initial state."""

    case = to_agent_case_context(bundle)
    inventory = _service_inventory(bundle)
    try:
        analysis = analyzer.analyze(case, bundle, inventory)
    except Exception as exc:
        failed_modalities = {
            modality for modality in Modality if modality in case.available_modalities
        }
        analysis = StaticAnalysisResult(
            statuses={
                modality: (
                    QueryStatus.TOOL_ERROR
                    if modality in failed_modalities
                    else QueryStatus.DATA_UNAVAILABLE
                )
                for modality in Modality
            },
            errors={
                modality: f"{type(exc).__name__}: static analysis failed"
                for modality in failed_modalities
            },
        )
    unknown_services = {finding.service for finding in analysis.findings} - inventory
    if unknown_services:
        raise ValueError(f"static analysis returned unknown services: {sorted(unknown_services)}")
    findings: dict[str, AnomalyFinding] = {
        finding.finding_id: finding for finding in analysis.findings
    }
    evidence_records = tuple(
        finding_to_evidence(finding, bundle.incident_start) for finding in analysis.findings
    )
    evidence = {record.evidence_id: record for record in evidence_records}
    query_statuses = {
        f"static|{modality.value}": status for modality, status in analysis.statuses.items()
    }
    return DiagnosisState(
        case=case,
        service_inventory=inventory,
        topology=build_observed_service_graph(evidence_records),
        initial_findings=findings,
        evidence=evidence,
        hypotheses={},
        tasks={},
        results=(),
        query_statuses=query_statuses,
        conflicts=(),
        max_supervisor_steps=max_supervisor_steps,
    )


def _record_for_case(
    manifest: DatasetManifest,
    case_id: str,
    dataset_root: Path,
) -> CaseRecord:
    manifest_case = next((item for item in manifest.cases if item.safe_case_id == case_id), None)
    if manifest_case is None:
        raise ValueError(f"case ID is not present in manifest: {case_id}")
    root = dataset_root.resolve()
    source = Path(manifest_case.source_locator).resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"case source is outside configured dataset root: {case_id}")
    return CaseRecord(
        safe_case_id=manifest_case.safe_case_id,
        source_locator=source,
        dataset_version=manifest.dataset_version,
        root_cause_service=manifest_case.root_cause_service,
        fault_type=manifest_case.fault_type,
        instance=1,
        checksums=manifest_case.checksums,
    )


def _source_references(record: CaseRecord) -> dict[Modality, str]:
    references: dict[Modality, str] = {}
    for modality in Modality:
        filename = next(
            (
                name
                for name in sorted(record.checksums)
                if name.casefold().startswith(modality.value)
            ),
            None,
        )
        if filename is None:
            continue
        references[modality] = (
            f"case:{record.safe_case_id}:file:{filename}:sha256:{record.checksums[filename]}"
        )
    return references


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, datetime):
        return value.isoformat()
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _flush_trace(runtime: TraceRuntime) -> None:
    try:
        runtime.flush()
    except Exception as exc:
        warnings.warn(f"Tracing flush failed: {exc}", RuntimeWarning, stacklevel=2)


def _diagnose_case(
    record: CaseRecord,
    settings: RuntimeSettings,
    output_dir: Path,
    run_id: str,
    static_analyzer: StaticAnalyzer | None = None,
) -> tuple[DiagnosisReport, Path]:
    bundle = load_re2_ob_case(record)
    runtime = create_trace_runtime(settings.tracing)
    state = _initialize_diagnosis_state(
        bundle,
        static_analyzer or RCAEvalMMBaroAnalyzer(),
        settings.mas.max_supervisor_steps,
    )
    runtime.event(
        "static_analysis.completed",
        {
            "finding_ids": sorted(state.initial_findings),
            "statuses": {
                signature: status.value
                for signature, status in sorted(state.query_statuses.items())
                if signature.startswith("static|")
            },
        },
        observation_type="evaluator",
    )
    metadata = RunMetadata(
        run_id=run_id,
        code_sha=os.getenv("POISE_RCA_CODE_SHA") or _git_sha(),
        config_fingerprint=sha256(settings.model_dump_json().encode()).hexdigest(),
        latency_ms=0,
    )
    graph = build_diagnosis_graph(
        GraphDependencies(
            hypothesis_model=_create_hypothesis_model(settings),
            supervisor_model=_create_supervisor_model(settings),
            worker_model=_create_worker_model(settings),
            max_supervisor_steps=settings.mas.max_supervisor_steps,
            tool_registry=ToolRegistry(
                bundle,
                timeout_seconds=settings.tools.per_tool_timeout_seconds,
                source_references=_source_references(record),
            ),
            tracer=runtime,
            run_metadata=metadata,
        )
    )
    started = perf_counter()
    try:
        with runtime.observation(
            "diagnosis-case",
            "chain",
            metadata={"case_id": record.safe_case_id, "run_id": run_id},
        ):
            config = (
                cast(RunnableConfig, {"callbacks": list(runtime.callbacks)})
                if runtime.callbacks
                else None
            )
            graph_input = cast(WorkflowState, state.model_dump(mode="python"))
            output = graph.invoke(graph_input, config=config)
            elapsed_ms = (perf_counter() - started) * 1000
            report = DiagnosisReport.model_validate(output["report"]).model_copy(
                update={"latency_ms": elapsed_ms, "trace_id": runtime.trace_id}
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            state_path = output_dir / "state.json"
            state_path.write_text(
                json.dumps(output, default=_json_default, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            report_path = output_dir / "report.json"
            write_report(report, report_path)
            try:
                runtime.set_trace_output(
                    {
                        "case_id": report.case_id,
                        "status": report.status.value,
                        "stop_reason": report.stop_reason,
                        "evidence_ids": sorted(output.get("evidence", {})),
                        "task_ids": sorted(output.get("tasks", {})),
                        "worker_conclusion_task_ids": [
                            item.task_id for item in report.worker_conclusions
                        ],
                        "hypothesis_states": {
                            item.hypothesis_id: item.status.value for item in report.candidates
                        },
                        "candidate_promotions": list(report.candidate_promotions),
                        "candidate_rejections": list(report.candidate_rejections),
                        "supervisor_steps": report.supervisor_steps,
                        "latency_ms": elapsed_ms,
                        "token_usage": report.token_usage,
                        "model_cost": report.model_cost,
                    }
                )
            except Exception as exc:
                warnings.warn(f"Tracing output failed: {exc}", RuntimeWarning, stacklevel=2)
            return report, report_path
    finally:
        _flush_trace(runtime)


@app.command()
def sample(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Create a deterministic sampled dataset manifest."""
    raw = _yaml_mapping(config)
    selection = raw.get("selection") or {}
    constraints = SamplingConstraints.model_validate(selection.get("constraints") or {})
    dataset_config = raw.get("dataset") or {}
    dataset_root = Path(
        os.getenv(
            "POISE_RCA_DATASET_ROOT",
            str(dataset_config.get("root") or "data/raw/rcaeval"),
        )
    )
    records = discover_re2_ob(dataset_root)
    if not records:
        raise typer.BadParameter(f"no RE2-OB cases found under {dataset_root}")
    version = str(dataset_config.get("version") or records[0].dataset_version)
    if version == "unspecified":
        version = records[0].dataset_version
    sampling = SamplingConfig(
        dataset_version=version,
        seed=int(selection["seed"]),
        size=int(selection["size"]),
        constraints=constraints,
    )
    manifest = create_stratified_manifest(records, sampling)
    destination = Path(str(raw["manifest"]))
    write_manifest(manifest, destination)
    typer.echo(str(destination))


@app.command()
def diagnose(
    case_id: Annotated[str, typer.Option(help="Case identifier from a manifest.")],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/default.yaml"),
) -> None:
    """Diagnose one case."""
    settings = _load_runtime_settings(config)
    dataset_manifest = DatasetManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    record = _record_for_case(dataset_manifest, case_id, settings.dataset.root)
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    output_dir = settings.results.root / run_id / case_id
    _, report_path = _diagnose_case(record, settings, output_dir, run_id)
    typer.echo(str(report_path))


@app.command()
def batch(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/default.yaml"),
) -> None:
    """Diagnose every case in a sampled manifest."""
    settings = _load_runtime_settings(config)
    dataset_manifest = DatasetManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    run_id = f"batch-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = settings.results.root / run_id
    reports: list[dict[str, str]] = []
    for manifest_case in dataset_manifest.cases:
        record = _record_for_case(
            dataset_manifest,
            manifest_case.safe_case_id,
            settings.dataset.root,
        )
        report, report_path = _diagnose_case(
            record,
            settings,
            run_dir / manifest_case.safe_case_id,
            run_id,
        )
        reports.append(
            {
                "case_id": report.case_id,
                "status": report.status.value,
                "report_path": str(report_path),
            }
        )
    summary = {
        "run_id": run_id,
        "case_count": len(reports),
        "status_counts": dict(Counter(item["status"] for item in reports)),
        "reports": reports,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    typer.echo(str(summary_path))


if __name__ == "__main__":
    app()
