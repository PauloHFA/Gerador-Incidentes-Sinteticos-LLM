"""Tests for QualityReporter, DatasetCardGenerator, and layered Config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from incidentforge.config import resolve_settings
from incidentforge.models import Incident, Language, Style, Topology
from incidentforge.report import DatasetCardGenerator, QualityReporter, generate_report


@pytest.mark.asyncio
async def test_quality_reporter_and_dataset_card(
    valid_incident: Incident, sample_topology: Topology, tmp_path: Path
):
    """QualityReporter computes distributions, validation metrics, and DatasetCard."""
    inc2 = valid_incident.model_copy(deep=True)
    inc2.id = "INC-2024-0002"
    inc2.metadata.language = Language.PT_BR
    inc2.metadata.style = Style.TERSE

    reporter = QualityReporter.from_incidents(
        [valid_incident, inc2],
        topology=sample_topology,
        rejected_count=1,
        first_pass_count=2,
        use_embeddings=False,
    )
    rep = reporter.generate_report()
    assert rep["summary"]["total_incidents"] == 2
    assert rep["distributions"]["by_language"]["pt-BR"] == 1
    assert rep["distributions"]["by_style"]["terse"] == 1
    assert rep["quality_metrics"]["rejection_rate"] == pytest.approx(1 / 3)
    assert rep["quality_metrics"]["first_pass_validation_rate"] == pytest.approx(2 / 3)
    assert rep["validation_metrics"]["schema_valid"] == 2
    assert rep["validation_metrics"]["semantic_valid"] == 2
    assert rep["validation_metrics"]["security_valid"] == 2

    md = reporter.to_markdown()
    assert "IncidentForge Dataset Quality Report" in md
    assert "By Language" in md
    parsed = json.loads(reporter.to_json())
    assert parsed["summary"]["total_incidents"] == 2

    paths = await generate_report(
        [valid_incident, inc2],
        tmp_path / "reports",
        format="both",
        topology=sample_topology,
        use_embeddings=False,
    )
    assert paths["markdown"].exists()
    assert paths["json"].exists()
    assert paths["dataset_card"].exists()
    assert "PauloHFA/IncidentForge" in paths["dataset_card"].read_text(encoding="utf-8")


def test_empty_reporter():
    """Empty incident list returns structured error dict and card."""
    reporter = QualityReporter([])
    assert reporter.generate_report() == {"error": "No incidents to report on"}
    assert "No incidents to report on" in reporter.to_markdown()
    card = DatasetCardGenerator([]).generate_card()
    assert "No incidents available" in card


def test_layered_config_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """CLI overrides > ENV > TOML > defaults."""
    toml_file = tmp_path / "incidentforge.toml"
    toml_file.write_text(
        '[llm]\nmodel = "toml-model"\ntemperature = 0.5\n[generation]\nconcurrency = 6\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("INCIDENTFORGE_MODEL", "env-model")
    monkeypatch.setenv("INCIDENTFORGE_SEED", "99")

    settings = resolve_settings(
        toml_path=toml_file,
        overrides={"model": "cli-model", "concurrency": None},
    )
    assert settings.model == "cli-model"
    assert settings.seed == 99
    assert settings.concurrency == 6
    assert settings.temperature == 0.5
