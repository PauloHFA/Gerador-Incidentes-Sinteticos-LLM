"""Tests for async generator orchestrator and resume/validation/budget controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from incidentforge.client import LLMClient, LLMResponse
from incidentforge.generator import (
    GenerationState,
    GeneratorConfig,
    IncidentGenerator,
    TokenBudgetExceeded,
    generate_incidents,
)
from incidentforge.models import (
    FailureCatalog,
    FailureCategory,
    FailureTemplate,
    GenerationTask,
    Incident,
    Language,
    Severity,
    Style,
    Topology,
)
from incidentforge.planner import PlannerConfig
from incidentforge.retry import RetryConfig


class ScriptedLLMClient(LLMClient):
    """Deterministic scripted LLM client for contract & orchestrator testing."""

    def __init__(self, responses_fn: Any) -> None:
        self.responses_fn = responses_fn
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response_format": response_format,
            }
        )
        return self.responses_fn(len(self.calls), prompt, system_prompt)

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str | None,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.generate(prompt, system_prompt, response_format={"schema": schema})

    async def close(self) -> None:
        self.closed = True


def _make_catalog() -> FailureCatalog:
    cats = list(FailureCategory)[:26]
    failures = []
    for i, cat in enumerate(cats):
        techs = ["nginx", "Go", "PostgreSQL", "Redis", "Python"] if i < 6 else ["CustomTech"]
        failures.append(
            FailureTemplate(
                category=cat,
                name=f"Failure {i}",
                description=f"Detailed failure description {i}",
                applicable_technologies=techs,
                typical_severities=[Severity.SEV_1, Severity.SEV_2],
                min_duration_minutes=5,
                max_duration_minutes=60,
            )
        )
    return FailureCatalog(version="1.0", failures=failures)


def _incident_payload_for_prompt(prompt: str, idx: int, *, invalid_pii: bool = False) -> str:
    service = "api-gateway"
    if "**Service:** auth-service" in prompt:
        service = "auth-service"
    elif "**Service:** auth-db" in prompt:
        service = "auth-db"
    elif "**Service:** rate-limit-cache" in prompt:
        service = "rate-limit-cache"

    severity = "SEV-2"
    for sev in ("SEV-1", "SEV-2", "SEV-3", "SEV-4"):
        if f"**Severity:** {sev}" in prompt:
            severity = sev
            break

    category = "Connection Leak"
    for cat in FailureCategory:
        if f"**Failure Category:** {cat.value}" in prompt or f"**Failure:** {cat.value}" in prompt:
            category = cat.value
            break

    tech_by_svc = {
        "api-gateway": "nginx",
        "auth-service": "PostgreSQL",
        "auth-db": "PostgreSQL",
        "rate-limit-cache": "Redis",
    }
    tech = tech_by_svc[service]
    extra = " Contact admin@realcorp.com" if invalid_pii else ""

    data = {
        "id": f"INC-2026-{idx:04d}",
        "service": service,
        "severity": severity,
        "outage_duration_minutes": 30,
        "failure_category": category,
        "sla_impact": {
            "availability_before": 0.9999,
            "availability_after": 0.9920,
            "p99_latency_ms": 4500,
        },
        "root_cause_analysis": (
            f"Incident {idx} in {service} caused by {tech} exhaustion under heavy load. "
            "Connection pools saturated across worker replicas because keepalive timeouts "
            "and pool reclamation were misconfigured during peak traffic." + extra
        ),
        "timeline": [
            {
                "timestamp": "2026-01-15T10:00:00Z",
                "event": "Anomaly begins",
                "source": tech,
                "severity": "SEV-4",
            },
            {
                "timestamp": "2026-01-15T10:05:00Z",
                "event": "Latency spikes",
                "source": service,
                "severity": "SEV-3",
            },
            {
                "timestamp": "2026-01-15T10:15:00Z",
                "event": "Alert paged on-call",
                "source": service,
                "severity": "SEV-2",
            },
            {
                "timestamp": "2026-01-15T10:25:00Z",
                "event": "Pool drained and restarted",
                "source": service,
                "severity": "SEV-2",
            },
            {
                "timestamp": "2026-01-15T10:30:00Z",
                "event": "Service stabilized",
                "source": service,
                "severity": "SEV-4",
            },
        ],
        "action_items": [
            {
                "action": f"Tune {tech} pool limits and timeouts",
                "owner": "sre-team",
                "priority": "P0",
            },
            {
                "action": "Add automated saturation alert thresholds",
                "owner": "platform-team",
                "priority": "P1",
            },
        ],
        "metadata": {
            "model": "gpt-4o-mini",
            "prompt_version": "1.0",
            "seed": 42 + idx,
            "generated_at": "2026-01-15T11:00:00Z",
            "schema_version": "1.0",
            "synthetic": True,
        },
    }
    return json.dumps(data)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_generator_concurrent_no_duplicate_workers(sample_topology: Topology, tmp_path: Path):
    """Concurrent workers must consume shared queue once each without duplicates."""
    catalog = _make_catalog()

    def respond(call_num: int, prompt: str, system_prompt: str | None) -> LLMResponse:
        return LLMResponse(
            content=_incident_payload_for_prompt(prompt, call_num),
            usage={"prompt_tokens": 100, "completion_tokens": 150, "total_tokens": 250},
            model="gpt-4o-mini",
            finish_reason="stop",
        )

    client = ScriptedLLMClient(respond)
    plan_cfg = PlannerConfig(count=6, seed=42)
    gen_cfg = GeneratorConfig(
        concurrency=4,
        state_dir=tmp_path / ".state",
        plan_config=plan_cfg,
    )
    gen = IncidentGenerator(
        sample_topology,
        gen_cfg,
        llm_client=client,
        failure_catalog=catalog,
    )
    await gen.initialize()
    incidents = await gen.generate()
    await gen.close()

    assert len(incidents) == 6
    assert len(client.calls) == 6
    assert client.closed is True
    assert (gen_cfg.state_dir / "incidents.jsonl").exists()
    assert (gen_cfg.state_dir / "state.jsonl").exists()
    assert gen._state is not None
    assert gen._state.first_pass_count == 6
    assert gen._state.total_tokens_input == 600
    assert gen._state.total_tokens_output == 900
    assert all(inc.metadata.language is not None for inc in incidents)
    assert all(inc.metadata.style is not None for inc in incidents)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_generator_validation_correction_retry_and_pii_rejection(
    sample_topology: Topology, tmp_path: Path
):
    """Invalid PII triggers corrective retry, and persistent invalid output saves to rejected.jsonl."""
    catalog = _make_catalog()
    call_counter = {"n": 0}

    def respond(call_num: int, prompt: str, system_prompt: str | None) -> LLMResponse:
        call_counter["n"] += 1
        # First incident: fails PII on attempt 1, succeeds on correction attempt 2
        # Second incident: always fails JSON parse -> goes to rejected.jsonl
        if "Correction Task" in prompt and call_counter["n"] == 2:
            content = _incident_payload_for_prompt(
                "**Service:** api-gateway **Severity:** SEV-2 **Failure Category:** Connection Leak",
                10,
                invalid_pii=False,
            )
        elif call_num == 1:
            content = _incident_payload_for_prompt(prompt, 1, invalid_pii=True)
        else:
            content = "not valid json at all"

        return LLMResponse(
            content=content,
            usage={"prompt_tokens": 50, "completion_tokens": 50},
            model="gpt-4o-mini",
            finish_reason="stop",
        )

    client = ScriptedLLMClient(respond)
    plan_cfg = PlannerConfig(
        count=2,
        seed=42,
        severities=[Severity.SEV_2],
        language_distribution={Language.EN: 1.0},
        style_distribution={Style.BLAMELESS_FORMAL: 1.0},
    )
    gen_cfg = GeneratorConfig(
        concurrency=1,
        max_validation_retries=1,
        retry_config=RetryConfig(max_attempts=1, base_wait=0.01),
        state_dir=tmp_path / ".state",
        plan_config=plan_cfg,
    )
    gen = IncidentGenerator(
        sample_topology,
        gen_cfg,
        llm_client=client,
        failure_catalog=catalog,
    )
    await gen.initialize()
    incidents = await gen.generate()
    await gen.close()

    # Check rejected.jsonl exists and recorded the permanently failing task
    rejected_file = gen_cfg.state_dir / "rejected.jsonl"
    assert rejected_file.exists()
    rejected_lines = [ln for ln in rejected_file.read_text(encoding="utf-8").splitlines() if ln]
    assert len(rejected_lines) >= 1
    assert isinstance(incidents, list)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_generator_resume_and_token_budget(sample_topology: Topology, tmp_path: Path):
    """Generator resumes from state.jsonl and enforces max_tokens_budget."""
    catalog = _make_catalog()

    def respond(call_num: int, prompt: str, system_prompt: str | None) -> LLMResponse:
        return LLMResponse(
            content=_incident_payload_for_prompt(prompt, call_num),
            usage={"prompt_tokens": 120, "completion_tokens": 120},
            model="gpt-4o-mini",
            finish_reason="stop",
        )

    state_dir = tmp_path / ".state"
    plan_cfg = PlannerConfig(count=4, seed=7)
    gen_cfg = GeneratorConfig(
        concurrency=1,
        max_tokens_budget=250,
        state_dir=state_dir,
        plan_config=plan_cfg,
        raise_on_budget_exceeded=True,
    )
    gen = IncidentGenerator(
        sample_topology,
        gen_cfg,
        llm_client=ScriptedLLMClient(respond),
        failure_catalog=catalog,
    )
    await gen.initialize()
    with pytest.raises(TokenBudgetExceeded):
        await gen.generate()
    await gen.close()

    # Resume with higher budget to finish remaining tasks
    gen_resume_cfg = GeneratorConfig(
        concurrency=2,
        max_tokens_budget=10000,
        state_dir=state_dir,
        resume=True,
        plan_config=plan_cfg,
    )
    client2 = ScriptedLLMClient(respond)
    gen2 = IncidentGenerator(
        sample_topology,
        gen_resume_cfg,
        llm_client=client2,
        failure_catalog=catalog,
    )
    await gen2.initialize()
    incidents = await gen2.generate()
    await gen2.close()

    assert len(incidents) == 4
    assert len(client2.calls) < 4  # resumed, did not regenerate already completed tasks


@pytest.mark.asyncio
async def test_generate_incidents_dry_run(sample_topology: Topology, tmp_path: Path):
    """Dry-run mode returns empty list without calling LLM."""
    incidents = await generate_incidents(
        topology=sample_topology,
        count=3,
        concurrency=2,
        dry_run=True,
        failure_catalog=_make_catalog(),
    )
    assert incidents == []


def test_generation_state_roundtrip(sample_topology: Topology, valid_incident: Incident):
    """GenerationState serializes and deserializes cleanly."""
    from incidentforge.planner import build_generation_plan

    plan = build_generation_plan(sample_topology, count=2, seed=42, catalog=_make_catalog())
    state = GenerationState(plan=plan, first_pass_count=1, total_attempts=1)
    key = "00000|api-gateway|Connection Leak|SEV-2|blameless_formal|en"
    state.completed_tasks[key] = valid_incident
    raw = state.to_json()
    restored = GenerationState.from_json(raw)
    assert restored.completed_count == 1
    assert restored.remaining_count == 1
    assert restored.first_pass_count == 1
    assert restored.elapsed_time >= 0.0
    assert isinstance(restored.plan.tasks[0], GenerationTask)
