from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from poise_rca.cli import _initialize_diagnosis_state, app
from poise_rca.data import discover_re2_ob, load_re2_ob_case
from poise_rca.domain import AnomalyFinding, Modality, QueryStatus, StaticAnalysisResult

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_case"


def test_cli_exposes_sample_diagnose_and_batch() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "sample" in result.stdout
    assert "diagnose" in result.stdout
    assert "batch" in result.stdout


class FakeStaticAnalyzer:
    def __init__(self) -> None:
        self.received_case: object | None = None
        self.received_bundle: object | None = None
        self.received_inventory: frozenset[str] | None = None

    def analyze(self, case, telemetry, service_inventory) -> StaticAnalysisResult:
        self.received_case = case
        self.received_bundle = telemetry
        self.received_inventory = service_inventory
        return StaticAnalysisResult(
            findings=(
                AnomalyFinding(
                    finding_id=str(uuid4()),
                    service="checkoutservice",
                    modality=Modality.METRICS,
                    signal="checkoutservice_cpu_usage",
                    observation="Checkout CPU changes near incident onset.",
                    source_reference=f"case:{case.case_id}:static:metric",
                    analyzer="fake-static",
                    score=0.9,
                    rank=1,
                ),
            ),
            statuses={
                Modality.METRICS: QueryStatus.SUCCESS_WITH_DATA,
                Modality.LOGS: QueryStatus.SUCCESS_EMPTY,
                Modality.TRACES: QueryStatus.DATA_UNAVAILABLE,
            },
        )


def test_static_findings_seed_graph_state_without_labels_or_source_path(
    tmp_path: Path,
) -> None:
    case_path = tmp_path / "re2ob_checkoutservice_delay_1"
    case_path.mkdir()
    for source in FIXTURE.iterdir():
        (case_path / source.name).write_bytes(source.read_bytes())
    record = discover_re2_ob(tmp_path)[0]
    bundle = load_re2_ob_case(record)
    analyzer = FakeStaticAnalyzer()

    state = _initialize_diagnosis_state(bundle, analyzer)

    assert analyzer.received_case == bundle.agent_view()
    assert analyzer.received_bundle == bundle
    assert analyzer.received_inventory
    assert state.initial_findings
    assert len(state.evidence) == 1
    evidence = next(iter(state.evidence.values()))
    assert evidence.attributes["analyzer"] == "fake-static"
    assert evidence.attributes["stage"] == "initialization"
    serialized = state.model_dump_json()
    assert "root_cause_service" not in serialized
    assert "fault_type" not in serialized
    assert str(record.source_locator) not in serialized
