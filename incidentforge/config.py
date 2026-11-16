"""Layered configuration loader for IncidentForge (CLI > ENV > TOML > defaults)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppSettings:
    """Resolved IncidentForge configuration across layers."""

    topology: Path = Path("configs/topology.yaml")
    failures: Path = Path("configs/failures.yaml")
    failures_mode: str = "extend"
    output_dir: Path = Path("output")
    state_dir: Path = Path(".incidentforge")
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: float = 120.0
    max_retries: int = 3
    concurrency: int = 4
    seed: int = 42
    max_tokens_budget: int | None = None
    similarity_threshold: float = 0.92
    minhash_threshold: float = 0.85
    export_formats: list[str] = field(default_factory=lambda: ["jsonl", "markdown"])


def load_toml_config(path: Path | None = None) -> dict[str, Any]:
    """Load TOML configuration from explicit path or default locations."""
    candidates = (
        [path] if path else [Path("incidentforge.toml"), Path("configs/incidentforge.toml")]
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            with candidate.open("rb") as f:
                data = tomllib.load(f)
            if isinstance(data, dict):
                return data
    return {}


def resolve_settings(
    toml_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppSettings:
    """Resolve configuration with precedence: overrides (CLI) > ENV > TOML > defaults."""
    settings = AppSettings()
    toml_data = load_toml_config(toml_path)

    # Flatten known TOML sections ([incidentforge], [llm], [generation], [dedup], [export])
    merged_toml: dict[str, Any] = {}
    for key, val in toml_data.items():
        if isinstance(val, dict):
            merged_toml.update(val)
        else:
            merged_toml[key] = val

    _apply_mapping(settings, merged_toml)

    # Environment variables layer
    env_map: dict[str, Any] = {}
    if v := os.getenv("INCIDENTFORGE_TOPOLOGY"):
        env_map["topology"] = Path(v)
    if v := os.getenv("INCIDENTFORGE_FAILURES"):
        env_map["failures"] = Path(v)
    if v := os.getenv("INCIDENTFORGE_FAILURES_MODE"):
        env_map["failures_mode"] = v
    if v := os.getenv("INCIDENTFORGE_OUTPUT_DIR"):
        env_map["output_dir"] = Path(v)
    if v := os.getenv("INCIDENTFORGE_STATE_DIR"):
        env_map["state_dir"] = Path(v)
    if v := os.getenv("INCIDENTFORGE_MODEL"):
        env_map["model"] = v
    if v := os.getenv("INCIDENTFORGE_BASE_URL"):
        env_map["base_url"] = v
    if v := (os.getenv("INCIDENTFORGE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        env_map["api_key"] = v
    if v := os.getenv("INCIDENTFORGE_TEMPERATURE"):
        env_map["temperature"] = float(v)
    if v := os.getenv("INCIDENTFORGE_MAX_TOKENS"):
        env_map["max_tokens"] = int(v)
    if v := os.getenv("INCIDENTFORGE_TIMEOUT"):
        env_map["timeout"] = float(v)
    if v := os.getenv("INCIDENTFORGE_MAX_RETRIES"):
        env_map["max_retries"] = int(v)
    if v := os.getenv("INCIDENTFORGE_CONCURRENCY"):
        env_map["concurrency"] = int(v)
    if v := os.getenv("INCIDENTFORGE_SEED"):
        env_map["seed"] = int(v)
    if v := os.getenv("INCIDENTFORGE_MAX_TOKENS_BUDGET"):
        env_map["max_tokens_budget"] = int(v)

    _apply_mapping(settings, env_map)

    if overrides:
        cleaned = {k: v for k, v in overrides.items() if v is not None}
        _apply_mapping(settings, cleaned)

    return settings


def _apply_mapping(settings: AppSettings, mapping: dict[str, Any]) -> None:
    """Apply dictionary values onto AppSettings with path/type coercion."""
    path_fields = {"topology", "failures", "output_dir", "state_dir"}
    for key, value in mapping.items():
        if not hasattr(settings, key) or value is None:
            continue
        if key in path_fields and not isinstance(value, Path):
            setattr(settings, key, Path(str(value)))
        elif key == "export_formats" and isinstance(value, str):
            setattr(settings, key, [part.strip() for part in value.split(",") if part.strip()])
        else:
            setattr(settings, key, value)
