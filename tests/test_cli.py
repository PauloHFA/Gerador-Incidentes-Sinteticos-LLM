"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from incidentforge.cli import app


class TestCLI:
    """Tests for CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_version(self):
        """Test version command."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "IncidentForge v" in result.output

    def test_help(self):
        """Test help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Synthetic post-mortem" in result.output
        assert "init" in result.output
        assert "plan" in result.output
        assert "generate" in result.output
        assert "validate" in result.output
        assert "dedup" in result.output
        assert "export" in result.output
        assert "report" in result.output
        assert "schema" in result.output

    def test_init_command(self):
        """Test init command creates sample files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "configs"
            result = self.runner.invoke(app, ["init", str(output_dir), "--force"])
            assert result.exit_code == 0
            assert "Created topology sample" in result.output
            assert "Created failures sample" in result.output

            # Check files exist
            assert (output_dir / "topology.yaml").exists()
            assert (output_dir / "failures.yaml").exists()

    def test_init_command_no_force(self):
        """Test init command without force on existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "configs"
            output_dir.mkdir(parents=True)
            (output_dir / "topology.yaml").write_text("existing")
            (output_dir / "failures.yaml").write_text("existing")

            result = self.runner.invoke(app, ["init", str(output_dir)])
            assert result.exit_code == 0
            assert "already exists" in result.output

    def test_plan_command(self):
        """Test plan command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample configs
            configs_dir = Path(tmpdir) / "configs"
            configs_dir.mkdir(parents=True)

            # Copy sample files
            import shutil

            sample_topology = Path("configs/topology_sample.yaml")
            sample_failures = Path("configs/failures_default.yaml")
            if sample_topology.exists():
                shutil.copy2(sample_topology, configs_dir / "topology.yaml")
            if sample_failures.exists():
                shutil.copy2(sample_failures, configs_dir / "failures.yaml")

            result = self.runner.invoke(
                app,
                [
                    "plan",
                    "--topology",
                    str(configs_dir / "topology.yaml"),
                    "--failures",
                    str(configs_dir / "failures.yaml"),
                    "--per-service",
                    "1",
                    "--seed",
                    "42",
                ],
            )
            assert result.exit_code == 0
            assert "Generation Plan" in result.output
            assert "Total Tasks:" in result.output

    def test_plan_command_dry_run(self):
        """Test plan command with dry run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configs_dir = Path(tmpdir) / "configs"
            configs_dir.mkdir(parents=True)

            import shutil

            sample_topology = Path("configs/topology_sample.yaml")
            sample_failures = Path("configs/failures_default.yaml")
            if sample_topology.exists():
                shutil.copy2(sample_topology, configs_dir / "topology.yaml")
            if sample_failures.exists():
                shutil.copy2(sample_failures, configs_dir / "failures.yaml")

            result = self.runner.invoke(
                app,
                [
                    "plan",
                    "--topology",
                    str(configs_dir / "topology.yaml"),
                    "--failures",
                    str(configs_dir / "failures.yaml"),
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "Dry run complete" in result.output

    def test_plan_command_output_json(self):
        """Test plan command with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configs_dir = Path(tmpdir) / "configs"
            configs_dir.mkdir(parents=True)

            import shutil

            sample_topology = Path("configs/topology_sample.yaml")
            sample_failures = Path("configs/failures_default.yaml")
            if sample_topology.exists():
                shutil.copy2(sample_topology, configs_dir / "topology.yaml")
            if sample_failures.exists():
                shutil.copy2(sample_failures, configs_dir / "failures.yaml")

            output_file = Path(tmpdir) / "plan.json"
            result = self.runner.invoke(
                app,
                [
                    "plan",
                    "--topology",
                    str(configs_dir / "topology.yaml"),
                    "--failures",
                    str(configs_dir / "failures.yaml"),
                    "--output",
                    str(output_file),
                ],
            )
            assert result.exit_code == 0
            assert output_file.exists()

            # Verify JSON content
            with output_file.open() as f:
                plan = json.load(f)
            assert "topology" in plan
            assert "failure_catalog" in plan
            assert "total_tasks" in plan
            assert "tasks" in plan

    def test_schema_command(self):
        """Test schema command."""
        result = self.runner.invoke(app, ["schema"])
        assert result.exit_code == 0
        # Should output JSON schema
        assert '"type": "object"' in result.output
        assert "Incident" in result.output or "id" in result.output

    def test_schema_command_output_file(self):
        """Test schema command with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "schema.json"
            result = self.runner.invoke(app, ["schema", "--output", str(output_file)])
            assert result.exit_code == 0
            assert output_file.exists()

            with output_file.open() as f:
                schema = json.load(f)
            assert schema["type"] == "object"

    def test_generate_command_missing_config(self):
        """Test generate command with missing config files."""
        runner = CliRunner()
        # Use non-existent config files
        result = runner.invoke(
            app,
            [
                "generate",
                "--topology",
                "nonexistent.yaml",
                "--failures",
                "nonexistent.yaml",
                "--concurrency",
                "0",
            ],
        )
        assert result.exit_code != 0
        assert "Error loading configuration" in result.output

    def test_validate_command_no_files(self):
        """Test validate command with no incident files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(app, ["validate", tmpdir])
            assert result.exit_code != 0
            assert "No JSONL files found" in result.output

    def test_dedup_command_no_files(self):
        """Test dedup command with no incident files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(app, ["dedup", tmpdir])
            assert result.exit_code != 0
            assert "No JSONL files found" in result.output

    def test_export_command_no_files(self):
        """Test export command with no incident files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(app, ["export", tmpdir])
            assert result.exit_code != 0
            assert "No JSONL files found" in result.output

    def test_report_command_no_files(self):
        """Test report command with no incident files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(app, ["report", tmpdir])
            assert result.exit_code != 0
            assert "No JSONL files found" in result.output

    def test_generate_dry_run_command(self, tmp_path: Path):
        """Test generate --dry-run runs planner and outputs token estimate."""
        result = self.runner.invoke(
            app,
            [
                "generate",
                "--topology",
                "configs/topology_sample.yaml",
                "--failures",
                "configs/failures_default.yaml",
                "--count",
                "4",
                "--dry-run",
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0
        assert "Dry run: 4 tasks planned" in result.output
        assert "Estimated tokens:" in result.output

    def test_generate_and_downstream_cli_commands(
        self, tmp_path: Path, valid_incident, monkeypatch
    ):
        """Test generate exports files, then validate/dedup/export/report work on them."""
        from incidentforge.generator import IncidentGenerator

        inc1 = valid_incident.model_copy(deep=True)
        inc2 = valid_incident.model_copy(deep=True)
        inc2.id = "INC-2024-0002"

        async def fake_generate(self):
            return [inc1, inc2]

        monkeypatch.setattr(IncidentGenerator, "generate", fake_generate)

        out_dir = tmp_path / "out"
        res_gen = self.runner.invoke(
            app,
            [
                "generate",
                "--topology",
                "configs/topology_sample.yaml",
                "--failures",
                "configs/failures_default.yaml",
                "--count",
                "2",
                "--output",
                str(out_dir),
                "--formats",
                "jsonl,markdown",
            ],
        )
        assert res_gen.exit_code == 0
        jsonl_file = out_dir / "incidents_dataset.jsonl"
        assert jsonl_file.exists()
        assert (out_dir / "markdown" / "INC-2024-0001.md").exists()

        # validate single JSONL file with --schema-only and full validation
        val_report = tmp_path / "val.json"
        res_val = self.runner.invoke(
            app,
            [
                "validate",
                str(jsonl_file),
                "--topology",
                "configs/topology_sample.yaml",
                "--output",
                str(val_report),
            ],
        )
        assert res_val.exit_code == 0
        assert val_report.exists()

        res_val_schema = self.runner.invoke(app, ["validate", str(jsonl_file), "--schema-only"])
        assert res_val_schema.exit_code == 0

        # dedup without embeddings
        dedup_dir = tmp_path / "dedup"
        res_dedup = self.runner.invoke(
            app,
            [
                "dedup",
                str(jsonl_file),
                "--output",
                str(dedup_dir),
                "--no-embeddings",
            ],
        )
        assert res_dedup.exit_code == 0
        assert (dedup_dir / "incidents_dataset.jsonl").exists()

        # export
        export_dir = tmp_path / "export"
        res_export = self.runner.invoke(
            app,
            [
                "export",
                str(jsonl_file),
                "--output",
                str(export_dir),
                "--formats",
                "jsonl,markdown",
            ],
        )
        assert res_export.exit_code == 0

        # report
        rep_dir = tmp_path / "reports"
        res_rep = self.runner.invoke(
            app,
            [
                "report",
                str(jsonl_file),
                "--output",
                str(rep_dir),
                "--topology",
                "configs/topology_sample.yaml",
            ],
        )
        assert res_rep.exit_code == 0
        assert (rep_dir / "quality_report.md").exists()
        assert (rep_dir / "quality_report.json").exists()
        assert (rep_dir / "DATASET_CARD.md").exists()

    def test_judge_command_mocked(self, tmp_path: Path, valid_incident, monkeypatch):
        """Test judge CLI command with mocked LLM client."""
        from incidentforge.client import LLMClient, LLMResponse

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(valid_incident.model_dump_json() + "\n", encoding="utf-8")
        out_json = tmp_path / "judge.json"

        class DummyJudgeClient(LLMClient):
            async def generate(self, prompt, system_prompt=None, response_format=None, **kwargs):
                return LLMResponse(
                    content=json.dumps(
                        {
                            "technical_realism": 5,
                            "causal_coherence": 4,
                            "actionability": 5,
                            "completeness": 5,
                            "consistency": 5,
                            "overall": 4.8,
                            "feedback": "Solid synthetic report",
                        }
                    )
                )

            async def generate_structured(self, prompt, system_prompt, schema, **kwargs):
                return await self.generate(prompt, system_prompt)

            async def close(self):
                pass

        monkeypatch.setattr("incidentforge.cli.create_llm_client", lambda cfg: DummyJudgeClient())

        res = self.runner.invoke(
            app,
            ["judge", str(jsonl_file), "--sample", "1", "--output", str(out_json)],
        )
        assert res.exit_code == 0
        assert "avg overall: 4.80/5" in res.output
        assert out_json.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
