"""Async generation orchestrator for IncidentForge."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles

from .client import LLMClient, LLMConfig, create_llm_client, generate_incident
from .models import (
    ArchitectureProfile,
    FailureCatalog,
    GenerationPlan,
    GenerationTask,
    Incident,
    Language,
    Service,
    Style,
    Topology,
)
from .planner import PlannerConfig, TaskPlanner, build_generation_plan
from .prompts import render_correction_prompt, render_incident_prompt
from .retry import RetryConfig, is_retryable_exception
from .validators import MultiLayerValidator, ValidationContext

logger = logging.getLogger(__name__)


@dataclass
class GenerationState:
    """State for generation resume capability."""

    plan: GenerationPlan
    completed_tasks: dict[str, Incident] = field(default_factory=dict)
    failed_tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected_tasks: list[dict[str, Any]] = field(default_factory=list)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    first_pass_count: int = 0
    total_attempts: int = 0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize state to JSON."""
        data = {
            "plan": self.plan.model_dump(),
            "completed_tasks": {k: v.model_dump() for k, v in self.completed_tasks.items()},
            "failed_tasks": self.failed_tasks,
            "rejected_tasks": self.rejected_tasks,
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "first_pass_count": self.first_pass_count,
            "total_attempts": self.total_attempts,
            "start_time": self.start_time,
            "last_update": self.last_update,
        }
        return json.dumps(data, default=str, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> GenerationState:
        """Deserialize state from JSON."""
        data = json.loads(json_str)
        plan = GenerationPlan.model_validate(data["plan"])
        completed = {k: Incident.model_validate(v) for k, v in data["completed_tasks"].items()}
        return cls(
            plan=plan,
            completed_tasks=completed,
            failed_tasks=data.get("failed_tasks", {}),
            rejected_tasks=data.get("rejected_tasks", []),
            total_tokens_input=data.get("total_tokens_input", 0),
            total_tokens_output=data.get("total_tokens_output", 0),
            first_pass_count=data.get("first_pass_count", 0),
            total_attempts=data.get("total_attempts", 0),
            start_time=data.get("start_time", time.time()),
            last_update=data.get("last_update", time.time()),
        )

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def completed_count(self) -> int:
        """Get number of completed tasks."""
        return len(self.completed_tasks)

    @property
    def remaining_count(self) -> int:
        """Get number of remaining tasks."""
        return max(0, self.plan.total_count - self.completed_count)


@dataclass
class GeneratorConfig:
    """Configuration for incident generator."""

    concurrency: int = 8
    max_tokens_budget: int | None = None
    max_validation_retries: int = 2
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    state_dir: Path = Path(".incidentforge")
    resume: bool = False
    dry_run: bool = False
    validate_output: bool = True
    plan_config: PlannerConfig | None = None
    raise_on_budget_exceeded: bool = False


class TokenBudgetExceeded(Exception):  # noqa: N818
    """Raised when token budget is exceeded."""

    def __init__(self, used: int, budget: int):
        self.used = used
        self.budget = budget
        super().__init__(f"Token budget exceeded: {used:,} / {budget:,}")


class IncidentGenerator:
    """Async incident generator with concurrency control and resume capability."""

    def __init__(
        self,
        topology: Topology,
        config: GeneratorConfig,
        llm_client: LLMClient | None = None,
        llm_config: LLMConfig | None = None,
        failure_catalog: FailureCatalog | None = None,
        plan_config: PlannerConfig | None = None,
    ):
        self.topology = topology
        self.config = config
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.failure_catalog = failure_catalog
        self.plan_config = plan_config or config.plan_config
        self._owned_client: LLMClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._state: GenerationState | None = None
        self._state_file: Path | None = None
        self._state_lock = asyncio.Lock()
        self._write_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._writer_task: asyncio.Task[None] | None = None
        self._shutdown = False
        self._budget_error: TokenBudgetExceeded | None = None

    def _build_plan(self) -> GenerationPlan:
        """Build generation plan using configured planner and catalog."""
        if self.plan_config is not None:
            return TaskPlanner(self.plan_config).build_plan(
                self.topology, catalog=self.failure_catalog
            )
        return build_generation_plan(
            self.topology,
            count=100,
            seed=42,
            catalog=self.failure_catalog,
        )

    def _get_client(self) -> LLMClient:
        """Get shared LLM client, lazily instantiating owned client if configured."""
        if self.llm_client is not None:
            return self.llm_client
        if self._owned_client is None and self.llm_config is not None:
            self._owned_client = create_llm_client(self.llm_config)
        if self._owned_client is None:
            raise ValueError("No LLM client available")
        return self._owned_client

    async def initialize(self) -> None:
        """Initialize generator (load state if resuming)."""
        self._shutdown = False
        self._budget_error = None
        self._semaphore = asyncio.Semaphore(max(1, self.config.concurrency))
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.config.state_dir / "state.jsonl"

        if self.config.resume and self._state_file.exists():
            await self._load_state()
        else:
            self._state = GenerationState(plan=self._build_plan())

        if not self.config.dry_run and (self._writer_task is None or self._writer_task.done()):
            self._writer_task = asyncio.create_task(self._background_writer())

    async def _load_state(self) -> None:
        """Load generation state from file."""
        assert self._state_file is not None
        try:
            async with aiofiles.open(self._state_file, encoding="utf-8") as f:
                content = await f.read()
            if content.strip():
                lines = [ln for ln in content.strip().split("\n") if ln.strip()]
                self._state = GenerationState.from_json(lines[-1])
                logger.info(
                    "Resumed state: %d/%d completed",
                    self._state.completed_count,
                    self._state.plan.total_count,
                )
            else:
                raise ValueError("Empty state file")
        except Exception as e:
            logger.warning("Failed to load state (%s). Starting fresh.", e)
            self._state = GenerationState(plan=self._build_plan())

    async def _save_state(self) -> None:
        """Save generation state to file."""
        if self._state is None:
            return
        self._state.last_update = time.time()
        state_json = self._state.to_json()
        await self._write_queue.put(("state", state_json))

    async def _background_writer(self) -> None:
        """Background task for async file writing."""
        assert self._state_file is not None
        state_file = self._state_file
        incidents_file = self.config.state_dir / "incidents.jsonl"
        rejected_file = self.config.state_dir / "rejected.jsonl"

        async with (
            aiofiles.open(state_file, "a", encoding="utf-8") as sf,
            aiofiles.open(incidents_file, "a", encoding="utf-8") as inf,
            aiofiles.open(rejected_file, "a", encoding="utf-8") as rf,
        ):
            while True:
                item_type, data = await self._write_queue.get()
                try:
                    if item_type == "__stop__":
                        break
                    if item_type == "state":
                        await sf.write(data + "\n")
                        await sf.flush()
                    elif item_type == "incident":
                        await inf.write(data + "\n")
                        await inf.flush()
                    elif item_type == "rejected":
                        await rf.write(data + "\n")
                        await rf.flush()
                finally:
                    self._write_queue.task_done()

    async def _stop_writer(self) -> None:
        """Flush and stop the background writer safely without deadlock."""
        if self._writer_task is not None and not self._writer_task.done():
            await self._write_queue.put(("__stop__", ""))
            await self._write_queue.join()
            await self._writer_task
            self._writer_task = None

    async def generate(self) -> list[Incident]:
        """Generate incidents according to plan."""
        if self._state is None:
            await self.initialize()
        assert self._state is not None

        if self.config.dry_run:
            logger.info(
                "Dry run: would generate %d incidents (est. %d tokens)",
                self._state.plan.total_count,
                self._state.plan.estimated_tokens,
            )
            return []

        remaining_tasks = self._get_remaining_tasks()
        if not remaining_tasks:
            logger.info("All tasks already completed")
            await self._stop_writer()
            return list(self._state.completed_tasks.values())

        logger.info(
            "Starting generation: %d remaining tasks (concurrency=%d)",
            len(remaining_tasks),
            self.config.concurrency,
        )

        await self._process_tasks(remaining_tasks)
        await self._save_state()
        await self._stop_writer()

        if self._budget_error is not None and self.config.raise_on_budget_exceeded:
            raise self._budget_error

        incidents = list(self._state.completed_tasks.values())
        logger.info(
            "Completed: %d incidents (%d in, %d out tokens, %.1fs)",
            len(incidents),
            self._state.total_tokens_input,
            self._state.total_tokens_output,
            self._state.elapsed_time,
        )
        return incidents

    def _get_remaining_tasks(self) -> list[tuple[int, GenerationTask]]:
        """Get indexed tasks that haven't been completed yet."""
        assert self._state is not None
        completed_keys = set(self._state.completed_tasks.keys())
        remaining: list[tuple[int, GenerationTask]] = []

        for idx, task in enumerate(self._state.plan.tasks):
            task_key = self._task_key(task, index=idx)
            if task_key not in completed_keys:
                remaining.append((idx, task))

        return remaining

    def _task_key(self, task: GenerationTask, index: int = 0) -> str:
        """Generate deterministic unique key for a planned task."""
        return (
            f"{index:05d}|{task.service_name}|{task.failure_category.value}"
            f"|{task.severity.value}|{task.style.value}|{task.language.value}"
        )

    async def _process_tasks(self, tasks: list[tuple[int, GenerationTask]]) -> None:
        """Process tasks via a shared queue so workers never duplicate work."""
        task_queue: asyncio.Queue[tuple[int, GenerationTask]] = asyncio.Queue()
        for item in tasks:
            task_queue.put_nowait(item)

        worker_count = min(max(1, self.config.concurrency), len(tasks))
        workers = [asyncio.create_task(self._worker(i, task_queue)) for i in range(worker_count)]
        await asyncio.gather(*workers)

    async def _worker(
        self, worker_id: int, task_queue: asyncio.Queue[tuple[int, GenerationTask]]
    ) -> None:
        """Worker coroutine consuming from the shared task queue."""
        assert self._semaphore is not None
        assert self._state is not None

        while not self._shutdown:
            try:
                idx, task = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                async with self._semaphore:
                    async with self._state_lock:
                        if self.config.max_tokens_budget is not None:
                            total_used = (
                                self._state.total_tokens_input + self._state.total_tokens_output
                            )
                            if total_used >= self.config.max_tokens_budget:
                                logger.warning(
                                    "[WORKER %d] Token budget reached (%d/%d), stopping",
                                    worker_id,
                                    total_used,
                                    self.config.max_tokens_budget,
                                )
                                self._budget_error = TokenBudgetExceeded(
                                    total_used, self.config.max_tokens_budget
                                )
                                self._shutdown = True
                                break

                    await self._process_task(worker_id, idx, task)
            finally:
                task_queue.task_done()

    async def _process_task(self, worker_id: int, index: int, task: GenerationTask) -> None:
        """Process a single generation task."""
        assert self._state is not None
        task_key = self._task_key(task, index=index)

        service = self.topology.get_service(task.service_name)
        if not service:
            logger.error("[WORKER %d] Service not found: %s", worker_id, task.service_name)
            return

        technologies = [t.name for t in service.technology]
        model_name = self.llm_config.model if self.llm_config else "gpt-4o-mini"
        prompt = render_incident_prompt(
            task=task,
            service=service,
            technologies=technologies,
            topology=self.topology,
            model=model_name,
            seed=self._state.plan.seed + index,
        )
        system_prompt = self._get_system_prompt(task)
        schema = Incident.model_json_schema()
        client = self._get_client()

        incident, attempts_used = await self._generate_with_validation(
            client=client,
            task=task,
            service=service,
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
        )

        should_checkpoint = False
        async with self._state_lock:
            self._state.total_attempts += attempts_used
            if incident is not None:
                self._state.completed_tasks[task_key] = incident
                if attempts_used == 1:
                    self._state.first_pass_count += 1
                if incident.metadata.tokens_input:
                    self._state.total_tokens_input += incident.metadata.tokens_input
                if incident.metadata.tokens_output:
                    self._state.total_tokens_output += incident.metadata.tokens_output
                should_checkpoint = len(self._state.completed_tasks) % 10 == 0
            else:
                self._state.failed_tasks[task_key] = {
                    "task": task.model_dump(),
                    "attempts": attempts_used,
                }

        if incident is not None:
            await self._write_queue.put(("incident", incident.model_dump_json()))
            logger.info("[WORKER %d] Completed %s -> %s", worker_id, task_key, incident.id)
        else:
            logger.warning("[WORKER %d] Failed %s", worker_id, task_key)

        if should_checkpoint:
            await self._save_state()

    async def _call_with_transport_retry(
        self,
        client: LLMClient,
        task: GenerationTask,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
    ) -> tuple[Incident | None, Any, str | None]:
        """Call LLM client with exponential backoff retry for transient errors."""
        retryer = self.config.retry_config.create_retryer()
        async for attempt in retryer:
            with attempt:
                incident, response, error = await generate_incident(
                    client=client,
                    task=task,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=schema,
                    use_structured=True,
                )
                return incident, response, error
        return None, None, "Retry attempts exhausted"

    async def _generate_with_validation(
        self,
        client: LLMClient,
        task: GenerationTask,
        service: Service,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
    ) -> tuple[Incident | None, int]:
        """Generate incident with validation retries."""
        assert self._state is not None
        last_error = "Unknown error"
        current_prompt = prompt
        max_attempts = self.config.max_validation_retries + 1

        for attempt in range(max_attempts):
            task_started = time.time()
            try:
                incident, response, error = await self._call_with_transport_retry(
                    client=client,
                    task=task,
                    prompt=current_prompt,
                    system_prompt=system_prompt,
                    schema=schema,
                )
            except Exception as e:
                last_error = str(e)
                if is_retryable_exception(e) and attempt < self.config.max_validation_retries:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                await self._save_rejected(task, current_prompt, last_error)
                return None, attempt + 1

            if error or incident is None:
                last_error = error or "Failed to parse incident from response"
                if attempt < self.config.max_validation_retries:
                    current_prompt = render_correction_prompt(
                        original_prompt=prompt,
                        error=last_error,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        task=task,
                    )
                    continue
                await self._save_rejected(task, current_prompt, last_error)
                return None, attempt + 1

            validation_error = self._validate_incident(incident, task, service)
            if validation_error:
                last_error = validation_error
                if attempt < self.config.max_validation_retries:
                    current_prompt = render_correction_prompt(
                        original_prompt=prompt,
                        error=validation_error,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        task=task,
                    )
                    continue
                await self._save_rejected(task, current_prompt, last_error)
                return None, attempt + 1

            incident.metadata.language = task.language
            incident.metadata.style = task.style
            incident.metadata.generation_time_ms = int((time.time() - task_started) * 1000)
            if response and response.usage:
                incident.metadata.tokens_input = response.usage.get("prompt_tokens", 0)
                incident.metadata.tokens_output = response.usage.get("completion_tokens", 0)

            return incident, attempt + 1

        await self._save_rejected(task, current_prompt, last_error)
        return None, max_attempts

    def _validate_incident(
        self, incident: Incident, task: GenerationTask, service: Service
    ) -> str | None:
        """Validate incident against task alignment and multi-layer validators."""
        if incident.service != task.service_name:
            return f"Incident service '{incident.service}' doesn't match task '{task.service_name}'"
        if incident.severity != task.severity:
            return f"Incident severity '{incident.severity}' doesn't match task '{task.severity}'"
        if incident.failure_category != task.failure_category:
            return (
                f"Incident failure category '{incident.failure_category}' "
                f"doesn't match task '{task.failure_category}'"
            )

        if self.config.validate_output:
            validator = MultiLayerValidator(
                ValidationContext(topology=self.topology, service=service)
            )
            errors = validator.get_all_errors(incident)
            if errors:
                return "; ".join(errors)

        return None

    async def _save_rejected(self, task: GenerationTask, prompt: str, error: str) -> None:
        """Save rejected incident record."""
        assert self._state is not None
        rejected = {
            "task": task.model_dump(),
            "prompt": prompt,
            "error": error,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        async with self._state_lock:
            self._state.rejected_tasks.append(rejected)
        await self._write_queue.put(("rejected", json.dumps(rejected, ensure_ascii=False)))

    def _get_system_prompt(self, task: GenerationTask) -> str:
        """Get system prompt based on task configuration."""
        base = (
            "You are an expert Site Reliability Engineer writing production incident post-mortems.\n"
            "Generate realistic, detailed incident reports in JSON format that conform exactly to the provided schema.\n\n"
            "Key principles:\n"
            "- Blameless culture: focus on systems, not individuals\n"
            "- Technical accuracy: use real technology names and failure modes\n"
            "- Actionable insights: action items should be specific and assignable\n"
            "- Consistency: timeline must be chronologically ordered and duration must match"
        )

        style_guidance = {
            Style.BLAMELESS_FORMAL: "Use formal, blameless language. Comprehensive detail.",
            Style.TERSE: "Be concise. Maximum 60% of detailed length. Essential facts only.",
            Style.DETAILED: "Provide extensive detail. Include context, debugging steps, and thorough analysis.",
        }

        lang_guidance = {
            Language.EN: "Write in English.",
            Language.PT_BR: "Escreva em português do Brasil. Mantenha chaves JSON em inglês.",
        }

        arch_guidance = {
            ArchitectureProfile.MICROSERVICES: "Context: Microservices architecture with distributed services.",
            ArchitectureProfile.MONOLITH_LEGACY: "Context: Legacy monolithic application with shared database.",
            ArchitectureProfile.SERVERLESS: "Context: Serverless architecture with managed services.",
        }

        return "\n\n".join(
            [
                base,
                style_guidance.get(task.style, ""),
                lang_guidance.get(task.language, ""),
                arch_guidance.get(task.architecture_profile, ""),
            ]
        )

    async def close(self) -> None:
        """Clean up resources."""
        self._shutdown = True
        await self._stop_writer()
        if self._owned_client is not None:
            await self._owned_client.close()
            self._owned_client = None
        if self.llm_client is not None:
            await self.llm_client.close()


async def generate_incidents(
    topology: Topology,
    count: int = 100,
    concurrency: int = 8,
    seed: int = 42,
    max_tokens_budget: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    llm_config: LLMConfig | None = None,
    failure_catalog: FailureCatalog | None = None,
) -> list[Incident]:
    """Convenience function to generate incidents."""
    plan_config = PlannerConfig(count=count, seed=seed, dry_run=dry_run)
    config = GeneratorConfig(
        concurrency=concurrency,
        max_tokens_budget=max_tokens_budget,
        resume=resume,
        dry_run=dry_run,
        plan_config=plan_config,
    )

    generator = IncidentGenerator(
        topology=topology,
        config=config,
        llm_config=llm_config,
        failure_catalog=failure_catalog,
        plan_config=plan_config,
    )
    await generator.initialize()

    try:
        return await generator.generate()
    finally:
        await generator.close()
