"""Model settings and factory for OpenAI-compatible chat endpoints."""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import yaml  # type: ignore[import-untyped]
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from poise_rca.agents.hypothesis_generator import (
    HypothesisBatch,
    HypothesisModel,
    HypothesisPromptInput,
)
from poise_rca.agents.supervisor import (
    SupervisorModel,
    SupervisorPlan,
    SupervisorPromptInput,
)
from poise_rca.agents.worker import (
    PromptedWorkerModel,
    WorkerAction,
    WorkerModel,
)


class ModelSettings(BaseModel):
    """Minimal model configuration; credentials remain outside diagnostic state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = "gpt-4.1-mini"
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)


def create_chat_model(settings: ModelSettings) -> BaseChatModel:
    """Create a chat model; callers apply the task-specific structured schema."""

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
    )


class MessageRunnable(Protocol):
    """Minimal structured runnable boundary used by the prompt adapter."""

    def invoke(self, input: object) -> object:
        """Invoke a structured model with LangChain messages."""


def _load_prompt(name: str) -> Mapping[str, object]:
    path = Path(__file__).with_name("prompts") / name
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{name} prompt must be a mapping")
    return loaded


def _prompt_text() -> tuple[str, str]:
    loaded = _load_prompt("hypothesis.yaml")
    system = loaded.get("system")
    repair = loaded.get("repair")
    if not isinstance(system, str) or not isinstance(repair, str):
        raise ValueError("hypothesis prompt requires system and repair text")
    return system.strip(), repair.strip()


class PromptedHypothesisModel:
    """Adapt label-safe structured inputs to the LangChain message interface."""

    def __init__(self, runnable: MessageRunnable) -> None:
        self._runnable = runnable
        self._system_prompt, self._repair_prompt = _prompt_text()

    def invoke(self, input: HypothesisPromptInput) -> object:
        messages: list[BaseMessage] = [SystemMessage(content=self._system_prompt)]
        if input.repair_error is not None:
            messages.append(
                SystemMessage(
                    content=f"{self._repair_prompt}\nValidation error: {input.repair_error}"
                )
            )
        payload = input.model_dump_json(indent=2, exclude={"repair_error"})
        messages.append(HumanMessage(content=f"Diagnostic input:\n{payload}"))
        return self._runnable.invoke(messages)


class PromptedSupervisorModel:
    """Adapt safe planning state to an independent structured model invocation."""

    def __init__(self, runnable: MessageRunnable) -> None:
        self._runnable = runnable
        system = _load_prompt("supervisor.yaml").get("system")
        if not isinstance(system, str):
            raise ValueError("supervisor prompt requires system text")
        self._system_prompt = system.strip()

    def invoke(self, input: SupervisorPromptInput) -> object:
        messages: list[BaseMessage] = [SystemMessage(content=self._system_prompt)]
        messages.append(HumanMessage(content=f"Planning input:\n{input.model_dump_json(indent=2)}"))
        return self._runnable.invoke(messages)


def create_hypothesis_model(settings: ModelSettings) -> HypothesisModel:
    """Create a schema-bound model that accepts the domain prompt contract."""

    chat_model = create_chat_model(settings)
    runnable = cast(MessageRunnable, chat_model.with_structured_output(HypothesisBatch))
    return PromptedHypothesisModel(runnable)


def create_supervisor_model(settings: ModelSettings) -> SupervisorModel:
    """Create a separately invoked schema-bound competitive-planning model."""

    chat_model = create_chat_model(settings)
    runnable = cast(MessageRunnable, chat_model.with_structured_output(SupervisorPlan))
    return PromptedSupervisorModel(runnable)


def create_worker_model(settings: ModelSettings) -> WorkerModel:
    """Create a schema-bound model invoked separately inside each worker subgraph."""

    chat_model = create_chat_model(settings)
    runnable = cast(MessageRunnable, chat_model.with_structured_output(WorkerAction))
    return PromptedWorkerModel(runnable)
