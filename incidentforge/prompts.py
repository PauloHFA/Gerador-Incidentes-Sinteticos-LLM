"""Prompt templates for IncidentForge."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from .models import (
    GenerationTask,
    Language,
    Service,
    Style,
    Topology,
)

# Jinja2 environment
PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION_PHRASE_RE = re.compile(
    r"(ignore\s+(?:previous|above|all)\s+(?:instructions|rules|prompts?)"
    r"|disregard\s+(?:previous|above|all)\s+(?:instructions|rules|prompts?)"
    r"|<\s*\|?\s*im_(?:start|end)\s*\|?\s*>)",
    re.IGNORECASE,
)


def sanitize_prompt_data(value: str, max_length: int = 240) -> str:
    """Sanitize untrusted topology/catalog text treated as data inside prompts."""
    cleaned = _CONTROL_CHAR_RE.sub("", value)
    cleaned = _INJECTION_PHRASE_RE.sub("[filtered]", cleaned)
    cleaned = " ".join(cleaned.splitlines()).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip() + "..."
    return cleaned


def _get_template(name: str) -> Template:
    """Get or create template."""
    template_path = PROMPTS_DIR / name
    if not template_path.exists():
        _create_default_templates()
    return _env.get_template(name)


def _create_default_templates() -> None:
    """Create default prompt templates."""
    # Incident generation template
    incident_template = """# Incident Generation Task

## Context
You are an expert Site Reliability Engineer writing a production incident post-mortem.
Generate a realistic, detailed incident report in JSON format.

**Architecture Profile:** {{ architecture_profile.value }}
**Language:** {{ language.value }}
**Style:** {{ style.value }}

## Service Information
**Service:** {{ service.name }}
**Team:** {{ service.team }}
**Region:** {{ service.region }}
**Replicas:** {{ service.replicas }}
**Language:** {{ service.language or 'N/A' }}
**Framework:** {{ service.framework or 'N/A' }}

**Technologies:**
{% for tech in technologies %}
- {{ tech }}
{% endfor %}

**SLO:**
- Availability Target: {{ service.slo.availability_target }}
- Latency P99: {{ service.slo.latency_p99_ms }}ms
{% if service.slo.throughput_rps %}- Throughput: {{ service.slo.throughput_rps }} rps{% endif %}

**Dependencies:**
{% for dep in service.dependencies %}
- {{ dep.target }} ({{ dep.traffic_percentage }}% traffic, critical: {{ dep.critical }})
{% endfor %}

## Failure Scenario
**Failure Category:** {{ failure_category.value }}
**Severity:** {{ severity.value }}

## Requirements
Generate a JSON object with the following structure (all fields required):

```json
{
  "id": "INC-YYYY-NNNN",
  "service": "{{ service.name }}",
  "severity": "{{ severity.value }}",
  "outage_duration_minutes": <integer > 0>,
  "failure_category": "{{ failure_category.value }}",
  "sla_impact": {
    "availability_before": <float 0-1>,
    "availability_after": <float 0-1>,
    "p99_latency_ms": <integer > 0>
  },
  "root_cause_analysis": "<string, minimum 150 characters, must mention at least one technology from the service>",
  "timeline": [
    {
      "timestamp": "ISO 8601 UTC",
      "event": "<description>",
      "source": "<component/service>",
      "severity": "<SEV-1 to SEV-4 or null>"
    }
    // Minimum 5 events in strictly increasing chronological order
  ],
  "action_items": [
    {
      "action": "<specific action>",
      "owner": "<team/person>",
      "priority": "<P0-P3>"
    }
    // 2-8 items
  ],
  "metadata": {
    "model": "{{ model }}",
    "prompt_version": "1.0",
    "seed": {{ seed }},
    "generated_at": "ISO 8601 UTC",
    "schema_version": "1.0",
    "synthetic": true
  }
}
```

## Style Guidelines
{% if style == Style.BLAMELESS_FORMAL %}
- Use formal, blameless language
- Focus on systemic factors, not individual errors
- Comprehensive detail with full context
{% elif style == Style.TERSE %}
- Be concise and direct
- Maximum 60% of detailed style length
- Essential facts only, no fluff
{% elif style == Style.DETAILED %}
- Provide extensive detail
- Include debugging steps, context, and thorough analysis
- Comprehensive timeline with technical specifics
{% endif %}

## Language
{% if language == Language.PT_BR %}
Escreva todo o conteúdo textual em português do Brasil.
Mantenha as chaves JSON em inglês.
{% else %}
Write all textual content in English.
Keep JSON keys in English.
{% endif %}

## Critical Rules
1. The `id` must follow pattern INC-YYYY-NNNN (use current year)
2. `outage_duration_minutes` must be consistent with timeline (within 10%)
3. Timeline must have at least 5 events in strictly increasing order
4. Root cause must mention at least one technology from the service: {{ technologies | join(', ') }}
5. Action items: 2-8 items, each with owner and priority P0-P3
6. SLA impact: availability_before > availability_after, realistic latency
7. All timestamps in ISO 8601 UTC format
8. Output ONLY valid JSON, no markdown, no extra text

Generate the incident now:"""

    (PROMPTS_DIR / "incident_v1.j2").write_text(incident_template, encoding="utf-8")

    # Correction template
    correction_template = """# Correction Task

The previous generation had validation errors. Please fix and regenerate.

## Original Task
**Service:** {{ task.service_name }}
**Failure:** {{ task.failure_category.value }}
**Severity:** {{ task.severity.value }}
**Style:** {{ task.style.value }}
**Language:** {{ task.language.value }}

## Validation Error
{{ error }}

## Attempt
This is attempt {{ attempt }} of {{ max_attempts }}.

## Instructions
Fix the specific validation error mentioned above. Common issues:
- JSON parsing: Ensure valid JSON only, no markdown fences
- Schema: All required fields present with correct types
- Timeline: Minimum 5 events, strictly increasing timestamps
- Duration: outage_duration_minutes within 10% of timeline span
- Root cause: Must mention service technology
- Action items: 2-8 items with owner and priority

Output ONLY the corrected JSON:"""

    (PROMPTS_DIR / "correction_v1.j2").write_text(correction_template, encoding="utf-8")

    # Judge template (for LLM-as-judge evaluation)
    judge_template = """# Incident Quality Evaluation

Evaluate the quality of this synthetic incident report on a scale of 1-5 for each criterion.

## Incident
{{ incident_json }}

## Evaluation Criteria
Rate each from 1 (poor) to 5 (excellent):

1. **Technical Realism**: Does the incident reflect realistic failure modes for the technologies?
2. **Causal Coherence**: Does the root cause logically explain the symptoms and timeline?
3. **Actionability**: Are action items specific, assignable, and likely to prevent recurrence?
4. **Completeness**: Are all required fields present with appropriate detail?
5. **Consistency**: Do timeline, duration, and impact metrics align?

## Output Format
```json
{
  "technical_realism": <1-5>,
  "causal_coherence": <1-5>,
  "actionability": <1-5>,
  "completeness": <1-5>,
  "consistency": <1-5>,
  "overall": <1-5>,
  "feedback": "<brief justification>"
}
```"""

    (PROMPTS_DIR / "judge_v1.j2").write_text(judge_template, encoding="utf-8")


def render_incident_prompt(
    task: GenerationTask,
    service: Service,
    technologies: list[str],
    topology: Topology,
    model: str = "gpt-4o-mini",
    seed: int = 42,
) -> str:
    """Render incident generation prompt."""
    template = _get_template("incident_v1.j2")
    sanitized_techs = [sanitize_prompt_data(t, max_length=80) for t in technologies]
    return str(
        template.render(
            task=task,
            service=service,
            technologies=sanitized_techs,
            topology=topology,
            architecture_profile=task.architecture_profile,
            language=task.language,
            style=task.style,
            failure_category=task.failure_category,
            severity=task.severity,
            model=model,
            seed=seed,
            Style=Style,
            Language=Language,
        )
    )


def render_correction_prompt(
    original_prompt: str,
    error: str,
    attempt: int,
    max_attempts: int = 3,
    task: GenerationTask | None = None,
) -> str:
    """Render correction prompt."""
    template = _get_template("correction_v1.j2")
    return str(
        template.render(
            original_prompt=original_prompt,
            error=sanitize_prompt_data(error, max_length=1000),
            attempt=attempt,
            max_attempts=max_attempts,
            task=task,
        )
    )


def render_judge_prompt(incident: dict[str, Any]) -> str:
    """Render LLM-as-judge evaluation prompt."""
    template = _get_template("judge_v1.j2")
    return str(template.render(incident_json=json.dumps(incident, indent=2, ensure_ascii=False)))
