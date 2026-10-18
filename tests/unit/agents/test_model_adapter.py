from datetime import UTC, datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from poise_rca.agents import HypothesisPromptInput, SupervisorPromptInput
from poise_rca.agents.model import PromptedHypothesisModel, PromptedSupervisorModel
from poise_rca.domain import CaseContext, Modality, QueryStatus
from poise_rca.topology import build_observed_service_graph


class FakeMessageRunnable:
    def __init__(self) -> None:
        self.inputs: list[object] = []

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        return {"hypotheses": []}


def _prompt_input(*, repair_error: str | None = None) -> HypothesisPromptInput:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return HypothesisPromptInput(
        case=CaseContext(
            case_id="re2-ob-0000000000000001",
            incident_start=start,
            incident_end=start + timedelta(minutes=5),
            initial_symptom="Elevated latency.",
            available_modalities=frozenset({Modality.METRICS}),
        ),
        topology=build_observed_service_graph([]),
        service_inventory=frozenset(),
        findings=(),
        analysis_statuses={Modality.METRICS: QueryStatus.SUCCESS_EMPTY},
        evidence=(),
        repair_error=repair_error,
    )


def test_prompted_model_converts_structured_input_to_langchain_messages() -> None:
    runnable = FakeMessageRunnable()
    model = PromptedHypothesisModel(runnable)

    result = model.invoke(_prompt_input())

    assert result == {"hypotheses": []}
    messages = runnable.inputs[0]
    assert isinstance(messages, list)
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "re2-ob-0000000000000001" in str(messages[1].content)


def test_prompted_model_includes_repair_instruction_without_raw_paths() -> None:
    runnable = FakeMessageRunnable()
    model = PromptedHypothesisModel(runnable)

    model.invoke(_prompt_input(repair_error="hypotheses field required"))

    serialized = repr(runnable.inputs[0])
    assert "previous response failed schema validation" in serialized.lower()
    assert "hypotheses field required" in serialized
    assert "source_locator" not in serialized


def test_prompted_supervisor_model_uses_separate_planning_prompt() -> None:
    runnable = FakeMessageRunnable()
    model = PromptedSupervisorModel(runnable)
    prompt_input = SupervisorPromptInput(
        hypotheses=(),
        evidence=(),
        topology=build_observed_service_graph([]),
        available_modalities=frozenset({Modality.TRACES}),
    )

    result = model.invoke(prompt_input)

    assert result == {"hypotheses": []}
    serialized = repr(runnable.inputs[0])
    assert "discriminating" in serialized.lower()
    assert "do not execute" in serialized.lower()
