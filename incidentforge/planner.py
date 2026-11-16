"""Deterministic task planner for IncidentForge."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .failures import FailureCatalog, load_failure_catalog
from .models import (
    ArchitectureProfile,
    GenerationPlan,
    GenerationTask,
    Language,
    Severity,
    Style,
    Topology,
)


@dataclass
class PlannerConfig:
    """Configuration for task planner."""

    count: int = 100
    seed: int = 42
    severities: list[Severity] | None = None
    severity_distribution: dict[Severity, float] | None = None
    style_distribution: dict[Style, float] | None = None
    language_distribution: dict[Language, float] | None = None
    architecture_profile: ArchitectureProfile = ArchitectureProfile.MICROSERVICES
    failures_mode: str = "extend"  # "extend" or "replace"
    user_failures_path: str | None = None
    dry_run: bool = False


class TaskPlanner:
    """Deterministic task planner using cartesian product with filtering."""

    def __init__(self, config: PlannerConfig):
        self.config = config
        self._rng = random.Random(config.seed)

    def build_plan(
        self, topology: Topology, catalog: FailureCatalog | None = None
    ) -> GenerationPlan:
        """
        Build generation plan from topology and failure catalog.

        Args:
            topology: System topology.
            catalog: Failure catalog (loads default if None).

        Returns:
            GenerationPlan with tasks.
        """
        if catalog is None:
            catalog = load_failure_catalog(
                self.config.user_failures_path, self.config.failures_mode
            )

        # Build technology map for compatibility filtering
        service_technologies = {}
        for service in topology.services:
            techs = [t.name for t in service.technology]
            service_technologies[service.name] = techs

        # Get applicable failures for each service
        service_failures = {}
        for service_name, techs in service_technologies.items():
            applicable = catalog.get_applicable_failures(techs)
            if applicable:
                service_failures[service_name] = [f.category for f in applicable]

        # Build cartesian product: services × failures × severities
        all_tasks = []
        severities = self.config.severities or list(Severity)

        for service_name, failure_categories in service_failures.items():
            for failure_category in failure_categories:
                for severity in severities:
                    task = GenerationTask(
                        service_name=service_name,
                        failure_category=failure_category,
                        severity=severity,
                        style=Style.BLAMELESS_FORMAL,  # Will be overridden per task
                        language=Language.EN,  # Will be overridden per task
                        architecture_profile=self.config.architecture_profile,
                    )
                    all_tasks.append(task)

        if not all_tasks:
            raise ValueError(
                "No valid task combinations found. Check topology and failure catalog compatibility."
            )

        # Apply severity distribution if specified
        if self.config.severity_distribution:
            all_tasks = self._apply_severity_distribution(all_tasks)

        # Shuffle deterministically
        self._rng.shuffle(all_tasks)

        # Select exact count
        if self.config.count > len(all_tasks):
            # Repeat with variation if needed
            selected = self._select_with_variation(all_tasks, self.config.count)
        else:
            selected = all_tasks[: self.config.count]

        # Apply style and language distributions
        selected = self._apply_style_language_distribution(selected)

        # Estimate tokens
        estimated_tokens = self._estimate_tokens(selected)

        return GenerationPlan(
            tasks=selected,
            total_count=len(selected),
            seed=self.config.seed,
            estimated_tokens=estimated_tokens,
        )

    def _apply_severity_distribution(self, tasks: list[GenerationTask]) -> list[GenerationTask]:
        """Apply severity distribution to tasks."""
        target_dist = self.config.severity_distribution
        if not target_dist:
            return tasks

        # Group by severity
        by_severity: dict[Severity, list[GenerationTask]] = {}
        for task in tasks:
            by_severity.setdefault(task.severity, []).append(task)

        # Calculate target counts
        total = len(tasks)
        target_counts = {sev: int(total * pct) for sev, pct in target_dist.items()}

        # Adjust to match total
        diff = total - sum(target_counts.values())
        if diff > 0 and target_counts:
            # Add to most common severity
            most_common = max(target_counts, key=lambda sev: target_counts[sev])
            target_counts[most_common] += diff

        # Select tasks per severity
        result = []
        for severity, count in target_counts.items():
            available = by_severity.get(severity, [])
            if len(available) >= count:
                result.extend(available[:count])
            else:
                # Repeat with variation
                result.extend(self._select_with_variation(available, count))

        return result

    def _apply_style_language_distribution(
        self, tasks: list[GenerationTask]
    ) -> list[GenerationTask]:
        """Apply style and language distributions."""
        # For now, assign based on distribution or default
        style_dist = self.config.style_distribution or {
            Style.BLAMELESS_FORMAL: 0.5,
            Style.DETAILED: 0.3,
            Style.TERSE: 0.2,
        }
        lang_dist = self.config.language_distribution or {
            Language.EN: 0.7,
            Language.PT_BR: 0.3,
        }

        # Convert to cumulative for selection
        style_items = list(style_dist.items())
        lang_items = list(lang_dist.items())

        for task in tasks:
            task.style = self._weighted_choice(style_items)
            task.language = self._weighted_choice(lang_items)

        return tasks

    def _weighted_choice(self, items: list[tuple[Any, float]]) -> Any:
        """Make weighted random choice."""
        r = self._rng.random()
        cumsum = 0.0
        for item, weight in items:
            cumsum += weight
            if r <= cumsum:
                return item
        return items[-1][0]

    def _select_with_variation(
        self,
        tasks: list[GenerationTask],
        count: int,
    ) -> list[GenerationTask]:
        """Select tasks with independent model copies when count exceeds available."""
        if not tasks:
            return []

        result: list[GenerationTask] = []
        while len(result) < count:
            for task in tasks:
                if len(result) >= count:
                    break
                result.append(task.model_copy(deep=True))

        return result[:count]

    def _estimate_tokens(self, tasks: list[GenerationTask]) -> int:
        """Estimate token cost for generation."""
        # Rough estimates per style
        tokens_per_style = {
            Style.BLAMELESS_FORMAL: 2500,
            Style.DETAILED: 3500,
            Style.TERSE: 1500,
        }

        total = 0
        for task in tasks:
            total += tokens_per_style.get(task.style, 2500)

        # Add overhead for prompt, retries, etc.
        return int(total * 1.3)


def create_planner(
    count: int = 100,
    seed: int = 42,
    severity_distribution: dict[Severity, float] | None = None,
    style_distribution: dict[Style, float] | None = None,
    language_distribution: dict[Language, float] | None = None,
    architecture_profile: ArchitectureProfile = ArchitectureProfile.MICROSERVICES,
    failures_mode: str = "extend",
    user_failures_path: str | None = None,
    dry_run: bool = False,
) -> TaskPlanner:
    """Create a task planner with the given configuration."""
    config = PlannerConfig(
        count=count,
        seed=seed,
        severity_distribution=severity_distribution,
        style_distribution=style_distribution,
        language_distribution=language_distribution,
        architecture_profile=architecture_profile,
        failures_mode=failures_mode,
        user_failures_path=user_failures_path,
        dry_run=dry_run,
    )
    return TaskPlanner(config)


def build_generation_plan(
    topology: Topology,
    count: int = 100,
    seed: int = 42,
    severity_distribution: dict[Severity, float] | None = None,
    style_distribution: dict[Style, float] | None = None,
    language_distribution: dict[Language, float] | None = None,
    architecture_profile: ArchitectureProfile = ArchitectureProfile.MICROSERVICES,
    failures_mode: str = "extend",
    user_failures_path: str | None = None,
    catalog: FailureCatalog | None = None,
) -> GenerationPlan:
    """
    Convenience function to build a generation plan.

    Args:
        topology: System topology.
        count: Number of incidents to generate.
        seed: Random seed for reproducibility.
        severity_distribution: Optional severity distribution.
        style_distribution: Optional style distribution.
        language_distribution: Optional language distribution.
        architecture_profile: Architecture profile.
        failures_mode: "extend" or "replace" for failure catalog.
        user_failures_path: Optional path to user failures.yaml.
        catalog: Optional pre-loaded failure catalog.

    Returns:
        GenerationPlan with tasks.
    """
    planner = create_planner(
        count=count,
        seed=seed,
        severity_distribution=severity_distribution,
        style_distribution=style_distribution,
        language_distribution=language_distribution,
        architecture_profile=architecture_profile,
        failures_mode=failures_mode,
        user_failures_path=user_failures_path,
    )
    return planner.build_plan(topology, catalog)


def print_plan_summary(plan: GenerationPlan) -> None:
    """Log a summary of the generation plan."""
    import logging

    log = logging.getLogger(__name__)
    log.info(
        "Generation plan summary: total=%d seed=%d estimated_tokens=%d",
        plan.total_count,
        plan.seed,
        plan.estimated_tokens,
    )
