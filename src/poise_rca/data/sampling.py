"""Deterministic, local-RNG sampling for smoke and prototype development sets."""

import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from poise_rca.data.manifest import DatasetManifest, ManifestCase
from poise_rca.data.models import CaseRecord

_FAULT_CATEGORY = {
    "cpu": "resource",
    "mem": "resource",
    "disk": "resource",
    "delay": "latency",
    "loss": "network",
    "socket": "network",
}


class SamplingConstraints(BaseModel):
    """Best-effort balancing rules for a deterministic sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples_per_fault_type: PositiveInt | None = None
    preferred_failure_categories: tuple[str, ...] = ()
    distinct_root_cause_services: bool = False
    balance_root_cause_services: bool = False


class SamplingConfig(BaseModel):
    """Inputs that completely determine one sampled manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str = Field(min_length=1)
    seed: int
    size: PositiveInt
    constraints: SamplingConstraints = SamplingConstraints()


def _pick_fault_strata(
    records: Sequence[CaseRecord],
    config: SamplingConfig,
    rng: random.Random,
) -> list[CaseRecord]:
    count = config.constraints.samples_per_fault_type
    if count is None:
        return []
    groups: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in records:
        groups[record.fault_type].append(record)

    selected: list[CaseRecord] = []
    service_counts: Counter[str] = Counter()
    for fault_type in sorted(groups):
        candidates = sorted(groups[fault_type], key=lambda item: item.safe_case_id)
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(f"not enough cases for fault type {fault_type!r}")
        for _ in range(count):
            if config.constraints.balance_root_cause_services:
                minimum = min(
                    service_counts[candidate.root_cause_service] for candidate in candidates
                )
                choice = next(
                    candidate
                    for candidate in candidates
                    if service_counts[candidate.root_cause_service] == minimum
                )
            else:
                choice = candidates[0]
            candidates.remove(choice)
            selected.append(choice)
            service_counts[choice.root_cause_service] += 1
    return selected


def _pick_categories(
    records: Sequence[CaseRecord],
    config: SamplingConfig,
    rng: random.Random,
) -> list[CaseRecord]:
    selected: list[CaseRecord] = []
    used_services: set[str] = set()
    for category in config.constraints.preferred_failure_categories:
        candidates = [
            record
            for record in records
            if _FAULT_CATEGORY.get(record.fault_type) == category and record not in selected
        ]
        candidates.sort(key=lambda item: item.safe_case_id)
        rng.shuffle(candidates)
        if not candidates:
            raise ValueError(f"no case available for failure category {category!r}")
        if config.constraints.distinct_root_cause_services:
            distinct = [
                candidate
                for candidate in candidates
                if candidate.root_cause_service not in used_services
            ]
            if distinct:
                candidates = distinct
        choice = candidates[0]
        selected.append(choice)
        used_services.add(choice.root_cause_service)
    return selected


def create_stratified_manifest(
    records: Sequence[CaseRecord],
    config: SamplingConfig,
) -> DatasetManifest:
    """Select cases reproducibly without mutating global random state."""

    eligible = [record for record in records if record.dataset_version == config.dataset_version]
    if len(eligible) < config.size:
        raise ValueError("not enough records for requested sample size")

    rng = random.Random(config.seed)
    if config.constraints.samples_per_fault_type is not None:
        selected = _pick_fault_strata(eligible, config, rng)
    elif config.constraints.preferred_failure_categories:
        selected = _pick_categories(eligible, config, rng)
    else:
        selected = rng.sample(sorted(eligible, key=lambda item: item.safe_case_id), config.size)

    if len(selected) > config.size:
        raise ValueError("sampling constraints select more cases than requested")
    if len(selected) < config.size:
        selected_ids = {record.safe_case_id for record in selected}
        remaining = [record for record in eligible if record.safe_case_id not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: config.size - len(selected)])

    constraints = config.constraints.model_dump()
    return DatasetManifest(
        dataset_version=config.dataset_version,
        seed=config.seed,
        constraints=constraints,
        cases=tuple(
            ManifestCase(
                safe_case_id=record.safe_case_id,
                source_locator=record.source_locator.as_posix(),
                root_cause_service=record.root_cause_service,
                fault_type=record.fault_type,
                checksums=record.checksums,
            )
            for record in selected
        ),
    )
