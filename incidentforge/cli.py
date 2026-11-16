"""IncidentForge CLI - Synthetic post-mortem and chaos scenario generator."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from .client import LLMConfig, TolerantJSONParser, create_llm_client
from .config import resolve_settings
from .dedup import DedupConfig, DiversityController
from .exporter import export_incidents
from .failures import FailureCatalogLoader, load_failure_catalog
from .generator import GeneratorConfig, IncidentGenerator
from .models import ArchitectureProfile, Incident, Language, Severity, Style, Topology
from .planner import PlannerConfig, TaskPlanner
from .prompts import render_judge_prompt
from .report import generate_report
from .topology import TopologyLoader
from .validators import (
    SchemaValidator,
    SecurityValidator,
    SemanticValidator,
    ValidationContext,
    ValidationResult,
)

app = typer.Typer(
    name="incidentforge",
    help="Synthetic post-mortem and chaos scenario generator for observability testing and incident response training.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def _parse_severities(raw: str) -> list[Severity]:
    """Parse comma-separated severity string."""
    severities: list[Severity] = []
    for item in raw.split(","):
        cleaned = item.strip().upper()
        if not cleaned:
            continue
        severities.append(Severity(cleaned))
    return severities or list(Severity)


def _parse_languages(raw: str) -> dict[Language, float]:
    """Parse comma-separated language distribution (e.g. 'en,pt-BR' or 'en:0.7,pt-BR:0.3')."""
    dist: dict[Language, float] = {}
    items = [part.strip() for part in raw.split(",") if part.strip()]
    for item in items:
        if ":" in item:
            lang_str, weight_str = item.split(":", 1)
            dist[Language(lang_str.strip())] = float(weight_str.strip())
        else:
            dist[Language(item)] = 1.0
    if not dist:
        return {Language.EN: 0.7, Language.PT_BR: 0.3}
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


def _parse_styles(raw: str) -> dict[Style, float]:
    """Parse comma-separated style distribution."""
    dist: dict[Style, float] = {}
    items = [part.strip() for part in raw.split(",") if part.strip()]
    for item in items:
        if ":" in item:
            style_str, weight_str = item.split(":", 1)
            dist[Style(style_str.strip())] = float(weight_str.strip())
        else:
            dist[Style(item)] = 1.0
    if not dist:
        return {
            Style.BLAMELESS_FORMAL: 0.5,
            Style.DETAILED: 0.3,
            Style.TERSE: 0.2,
        }
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


def _find_jsonl_files(input_path: Path) -> list[Path]:
    """Resolve JSONL file(s) from either a single file or a directory."""
    if input_path.is_file():
        return [input_path] if input_path.suffix == ".jsonl" else []
    if input_path.is_dir():
        return sorted(input_path.rglob("*.jsonl"))
    return []


def _load_incidents_from_path(input_path: Path) -> list[Incident]:
    """Load Incident objects from JSONL file or directory."""
    jsonl_files = _find_jsonl_files(input_path)
    if not jsonl_files:
        console.print(f"[red]No JSONL files found in {input_path}[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(jsonl_files)} JSONL file(s)")
    incidents: list[Incident] = []
    for file_path in jsonl_files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    incidents.append(Incident.model_validate(data))
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to parse line in {file_path}: {e}[/yellow]"
                    )
    return incidents


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from . import __version__

        console.print(f"IncidentForge v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = False,
) -> None:
    """IncidentForge - Generate synthetic incident reports from system topology."""


@app.command()
def init(
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory to create sample configuration files"),
    ] = Path("configs"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing files"),
    ] = False,
) -> None:
    """Create sample topology and failure catalog configuration files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_path = output_dir / "topology.yaml"
    failures_path = output_dir / "failures.yaml"

    if topology_path.exists() and not force:
        console.print(f"[yellow]Topology file already exists: {topology_path}[/yellow]")
    else:
        sample_topology = Path(__file__).parent.parent / "configs" / "topology_sample.yaml"
        if sample_topology.exists():
            shutil.copy2(sample_topology, topology_path)
            console.print(f"[green]Created topology sample: {topology_path}[/green]")
        else:
            console.print("[red]Sample topology not found in package[/red]")

    if failures_path.exists() and not force:
        console.print(f"[yellow]Failures file already exists: {failures_path}[/yellow]")
    else:
        sample_failures = Path(__file__).parent.parent / "configs" / "failures_default.yaml"
        if sample_failures.exists():
            shutil.copy2(sample_failures, failures_path)
            console.print(f"[green]Created failures sample: {failures_path}[/green]")
        else:
            console.print("[red]Sample failures not found in package[/red]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"  1. Edit {topology_path} to match your system topology")
    console.print(f"  2. Edit {failures_path} to customize failure categories")
    console.print("  3. Run 'incidentforge plan' to preview generation tasks")
    console.print("  4. Run 'incidentforge generate' to create incidents")


@app.command()
def plan(
    topology: Annotated[
        Path,
        typer.Option("--topology", "-t", help="Path to topology YAML file"),
    ] = Path("configs/topology.yaml"),
    failures: Annotated[
        Path | None,
        typer.Option("--failures", "-f", help="Path to failure catalog YAML file"),
    ] = Path("configs/failures.yaml"),
    failures_mode: Annotated[
        str,
        typer.Option("--failures-mode", help="Failure catalog mode: extend or replace"),
    ] = "extend",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file for plan (JSON)"),
    ] = None,
    count: Annotated[
        int | None,
        typer.Option("--count", help="Exact number of incidents to plan"),
    ] = None,
    per_service: Annotated[
        int,
        typer.Option("--per-service", "-n", help="Incidents per service (used if --count omitted)"),
    ] = 3,
    per_category: Annotated[
        int,
        typer.Option("--per-category", "-c", help="Incidents per failure category (reference)"),
    ] = 2,
    severities: Annotated[
        str,
        typer.Option(
            "--severities", "-s", help="Comma-separated severities (SEV-1,SEV-2,SEV-3,SEV-4)"
        ),
    ] = "SEV-1,SEV-2,SEV-3",
    lang: Annotated[
        str,
        typer.Option("--lang", help="Language distribution (e.g. en:0.7,pt-BR:0.3)"),
    ] = "en:0.7,pt-BR:0.3",
    style: Annotated[
        str,
        typer.Option("--style", help="Style distribution (blameless_formal,terse,detailed)"),
    ] = "blameless_formal:0.5,detailed:0.3,terse:0.2",
    architecture_profile: Annotated[
        ArchitectureProfile,
        typer.Option("--architecture-profile", help="Architecture profile"),
    ] = ArchitectureProfile.MICROSERVICES,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = 42,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-d", help="Show plan without executing"),
    ] = True,
) -> None:
    """Preview generation plan without executing."""
    console.print("[bold]Loading configuration...[/bold]")

    try:
        topology_obj = TopologyLoader().load(topology)
        failures_obj = (
            load_failure_catalog(failures, mode=failures_mode)
            if failures is not None and failures.exists()
            else FailureCatalogLoader().default_catalog
        )
        if failures is not None and not failures.exists():
            raise FileNotFoundError(f"Failures file not found: {failures}")
        sev_list = _parse_severities(severities)
        lang_dist = _parse_languages(lang)
        style_dist = _parse_styles(style)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1) from e

    resolved_count = (
        count if count is not None else max(1, per_service * len(topology_obj.services))
    )
    config = PlannerConfig(
        count=resolved_count,
        seed=seed,
        severities=sev_list,
        style_distribution=style_dist,
        language_distribution=lang_dist,
        architecture_profile=architecture_profile,
        failures_mode=failures_mode,
        dry_run=dry_run,
    )

    planner = TaskPlanner(config)
    gen_plan = planner.build_plan(topology_obj, failures_obj)
    tasks = gen_plan.tasks

    console.print("\n[bold]Generation Plan[/bold]")
    console.print(f"  Topology: {topology_obj.version} ({len(topology_obj.services)} services)")
    console.print(
        f"  Failure Catalog: {failures_obj.version} ({len(failures_obj.failures)} categories)"
    )
    console.print(f"  Total Tasks: {len(tasks)}")
    console.print(f"  Estimated Tokens: {gen_plan.estimated_tokens:,}")
    console.print(f"  Seed: {seed}")

    table = Table(title="Task Distribution")
    table.add_column("Service", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Severity", style="yellow")
    table.add_column("Count", justify="right", style="green")

    dist = Counter((t.service_name, t.failure_category.value, t.severity.value) for t in tasks)
    for (svc, cat, sev), item_count in sorted(dist.items()):
        table.add_row(svc, cat, sev, str(item_count))

    console.print(table)

    if output:
        plan_data = {
            "topology": topology_obj.version,
            "failure_catalog": failures_obj.version,
            "total_tasks": len(tasks),
            "estimated_tokens": gen_plan.estimated_tokens,
            "seed": seed,
            "config": dataclasses.asdict(config),
            "tasks": [t.model_dump() for t in tasks],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan_data, indent=2, default=str), encoding="utf-8")
        console.print(f"\n[green]Plan saved to: {output}[/green]")

    if dry_run:
        console.print(
            "\n[yellow]Dry run complete. Use 'incidentforge generate' to execute.[/yellow]"
        )


@app.command()
def generate(
    topology: Annotated[
        Path | None,
        typer.Option("--topology", "-t", help="Path to topology YAML file"),
    ] = None,
    failures: Annotated[
        Path | None,
        typer.Option("--failures", "-f", help="Path to failure catalog YAML file"),
    ] = None,
    failures_mode: Annotated[
        str,
        typer.Option("--failures-mode", help="Failure catalog mode: extend or replace"),
    ] = "extend",
    output_dir: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for generated incidents"),
    ] = None,
    count: Annotated[
        int | None,
        typer.Option("--count", help="Exact number of incidents to generate"),
    ] = None,
    per_service: Annotated[
        int,
        typer.Option("--per-service", "-n", help="Incidents per service (used if --count omitted)"),
    ] = 3,
    per_category: Annotated[
        int,
        typer.Option("--per-category", "-c", help="Incidents per failure category (reference)"),
    ] = 2,
    severities: Annotated[
        str,
        typer.Option("--severities", "-s", help="Comma-separated severities"),
    ] = "SEV-1,SEV-2,SEV-3",
    lang: Annotated[
        str,
        typer.Option("--lang", help="Language distribution (e.g. en:0.7,pt-BR:0.3)"),
    ] = "en:0.7,pt-BR:0.3",
    style: Annotated[
        str,
        typer.Option("--style", help="Style distribution (blameless_formal,terse,detailed)"),
    ] = "blameless_formal:0.5,detailed:0.3,terse:0.2",
    architecture_profile: Annotated[
        ArchitectureProfile,
        typer.Option("--architecture-profile", help="Architecture profile"),
    ] = ArchitectureProfile.MICROSERVICES,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Max concurrent LLM requests"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LLM model name"),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="LLM API base URL (for vLLM, Ollama, NIM, etc.)"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="LLM API key (or set INCIDENTFORGE_API_KEY env var)"),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="LLM temperature"),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Max tokens per request"),
    ] = None,
    max_tokens_budget: Annotated[
        int | None,
        typer.Option("--max-tokens-budget", help="Total token budget across batch"),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Request timeout in seconds"),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option("--max-retries", help="Max retry attempts"),
    ] = None,
    validate: Annotated[
        bool,
        typer.Option("--validate/--no-validate", help="Run validation pipeline"),
    ] = True,
    dedup: Annotated[
        bool,
        typer.Option("--dedup/--no-dedup", help="Run deduplication"),
    ] = True,
    export_formats: Annotated[
        str | None,
        typer.Option("--formats", help="Comma-separated export formats (jsonl,markdown,parquet)"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume from previous state"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-d", help="Preview plan and token estimate without calling LLM"),
    ] = False,
    config_file: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to incidentforge.toml"),
    ] = None,
) -> None:
    """Generate synthetic incident reports."""
    console.print("[bold]IncidentForge - Generating Incidents[/bold]")

    settings = resolve_settings(
        toml_path=config_file,
        overrides={
            "topology": topology,
            "failures": failures,
            "failures_mode": failures_mode,
            "output_dir": output_dir,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
            "concurrency": concurrency,
            "seed": seed,
            "max_tokens_budget": max_tokens_budget,
            "export_formats": export_formats,
        },
    )

    try:
        topology_obj = TopologyLoader().load(settings.topology)
        failures_obj = load_failure_catalog(settings.failures, mode=settings.failures_mode)
        sev_list = _parse_severities(severities)
        lang_dist = _parse_languages(lang)
        style_dist = _parse_styles(style)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1) from e

    resolved_count = (
        count if count is not None else max(1, per_service * len(topology_obj.services))
    )
    plan_config = PlannerConfig(
        count=resolved_count,
        seed=settings.seed,
        severities=sev_list,
        style_distribution=style_dist,
        language_distribution=lang_dist,
        architecture_profile=architecture_profile,
        failures_mode=settings.failures_mode,
        dry_run=dry_run,
    )

    llm_config = LLMConfig(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )

    gen_config = GeneratorConfig(
        concurrency=max(1, settings.concurrency),
        max_tokens_budget=settings.max_tokens_budget,
        state_dir=settings.state_dir,
        resume=resume,
        dry_run=dry_run,
        validate_output=validate,
        plan_config=plan_config,
    )

    generator = IncidentGenerator(
        topology=topology_obj,
        config=gen_config,
        llm_config=llm_config,
        failure_catalog=failures_obj,
        plan_config=plan_config,
    )

    async def _run() -> list[Incident]:
        try:
            await generator.initialize()
            return await generator.generate()
        finally:
            await generator.close()

    try:
        incidents = asyncio.run(_run())
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Generation interrupted by user[/yellow]")
        raise typer.Exit(130) from exc
    except Exception as e:
        console.print(f"\n[red]Generation failed: {e}[/red]")
        raise typer.Exit(1) from e

    if dry_run:
        plan_preview = TaskPlanner(plan_config).build_plan(topology_obj, failures_obj)
        console.print(f"[yellow]Dry run: {plan_preview.total_count} tasks planned[/yellow]")
        console.print(f"[yellow]Estimated tokens: {plan_preview.estimated_tokens:,}[/yellow]")
        return

    if dedup and incidents:
        dedup_config = DedupConfig(
            similarity_threshold=settings.similarity_threshold,
            minhash_threshold=settings.minhash_threshold,
            use_embeddings=False,
            use_minhash=True,
        )
        controller = DiversityController(dedup_config)
        unique_incidents: list[Incident] = []
        for inc in incidents:
            is_dup, _, _ = controller.check_duplicate(inc)
            if not is_dup:
                controller.add(inc)
                unique_incidents.append(inc)
        removed = len(incidents) - len(unique_incidents)
        if removed:
            console.print(f"[yellow]Deduplication removed {removed} near-duplicate(s)[/yellow]")
        incidents = unique_incidents

    fmt_set = {f.lower() for f in settings.export_formats}
    if incidents:
        paths = asyncio.run(
            export_incidents(
                incidents,
                settings.output_dir,
                export_markdown="markdown" in fmt_set,
                export_jsonl="jsonl" in fmt_set,
                export_parquet="parquet" in fmt_set,
            )
        )
        console.print(f"\n[green]Successfully generated {len(incidents)} incidents![/green]")
        console.print(f"Output directory: {settings.output_dir}")
        for fmt, path in paths.items():
            console.print(f"  {fmt}: {path}")
    else:
        console.print("\n[yellow]No valid incidents were generated.[/yellow]")


@app.command()
def validate(
    input_path: Annotated[
        Path,
        typer.Argument(help="Directory or JSONL file containing generated incidents"),
    ] = Path("output"),
    topology: Annotated[
        Path,
        typer.Option("--topology", "-t", help="Path to topology YAML file"),
    ] = Path("configs/topology.yaml"),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file for validation report"),
    ] = None,
    schema_only: Annotated[
        bool,
        typer.Option("--schema-only", help="Only validate JSON schema"),
    ] = False,
    semantic: Annotated[
        bool,
        typer.Option("--semantic/--no-semantic", help="Run semantic validation"),
    ] = True,
    security: Annotated[
        bool,
        typer.Option("--security/--no-security", help="Run security/PII validation"),
    ] = True,
) -> None:
    """Validate generated incidents."""
    console.print("[bold]Validating Incidents[/bold]")
    incidents = _load_incidents_from_path(input_path)
    console.print(f"Loaded {len(incidents)} incidents")

    run_semantic = semantic and not schema_only
    run_security = security and not schema_only

    topology_obj: Topology | None = None
    if run_semantic or run_security:
        try:
            topology_obj = TopologyLoader().load(topology)
        except Exception as e:
            console.print(f"[red]Error loading topology: {e}[/red]")
            raise typer.Exit(1) from e

    console.print(
        f"Running validation (schema=True, semantic={run_semantic}, security={run_security})"
    )

    all_results: list[ValidationResult] = []
    for incident in incidents:
        schema_res = SchemaValidator.validate(incident)
        all_results.append(schema_res)
        if not schema_res.valid or schema_only:
            continue

        assert topology_obj is not None
        service = topology_obj.get_service(incident.service)
        if service is None:
            all_results.append(
                ValidationResult(
                    valid=False,
                    errors=[f"Service '{incident.service}' not found in topology"],
                    layer="semantic",
                )
            )
            continue

        ctx = ValidationContext(topology=topology_obj, service=service)
        if run_semantic:
            all_results.append(SemanticValidator(ctx).validate(incident))
        if run_security:
            all_results.append(SecurityValidator(ctx).validate(incident))

    passed = sum(1 for r in all_results if r.valid)
    failed = len(all_results) - passed

    console.print("\n[bold]Validation Results[/bold]")
    console.print(f"  Total checks: {len(all_results)}")
    console.print(f"  Passed: {passed}")
    console.print(f"  Failed: {failed}")

    if output:
        report = {
            "total": len(all_results),
            "passed": passed,
            "failed": failed,
            "results": [r.model_dump() for r in all_results],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        console.print(f"\n[green]Validation report saved to: {output}[/green]")

    if failed > 0:
        raise typer.Exit(1)


@app.command()
def dedup(
    input_path: Annotated[
        Path,
        typer.Argument(help="Directory or JSONL file containing incident JSONL files"),
    ] = Path("output"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for deduplicated incidents"),
    ] = Path("output_dedup"),
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Similarity threshold (0-1)"),
    ] = 0.92,
    use_embeddings: Annotated[
        bool,
        typer.Option("--embeddings/--no-embeddings", help="Use embedding-based deduplication"),
    ] = True,
    use_minhash: Annotated[
        bool,
        typer.Option("--minhash/--no-minhash", help="Use MinHash-based deduplication"),
    ] = True,
) -> None:
    """Deduplicate incidents using semantic similarity."""
    console.print("[bold]Deduplicating Incidents[/bold]")
    incidents = _load_incidents_from_path(input_path)
    console.print(f"Loaded {len(incidents)} incidents")

    config = DedupConfig(
        similarity_threshold=threshold,
        use_embeddings=use_embeddings,
        use_minhash=use_minhash,
    )
    controller = DiversityController(config)

    unique_incidents: list[Incident] = []
    duplicates: list[tuple[Incident, str | None, float]] = []

    for incident in incidents:
        is_dup, dup_id, sim = controller.check_duplicate(incident)
        if is_dup:
            duplicates.append((incident, dup_id, sim))
        else:
            controller.add(incident)
            unique_incidents.append(incident)

    console.print("\n[bold]Deduplication Results[/bold]")
    console.print(f"  Original: {len(incidents)}")
    console.print(f"  Unique: {len(unique_incidents)}")
    console.print(f"  Duplicates removed: {len(duplicates)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = asyncio.run(
        export_incidents(
            unique_incidents,
            output_dir,
            export_markdown=False,
            export_jsonl=True,
            export_parquet=False,
        )
    )
    console.print(f"\n[green]Deduplicated incidents saved to: {output_dir}[/green]")
    for fmt, path in paths.items():
        console.print(f"  {fmt}: {path}")


@app.command()
def export(
    input_path: Annotated[
        Path,
        typer.Argument(help="Directory or JSONL file containing incident JSONL files"),
    ] = Path("output"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("export"),
    formats: Annotated[
        str,
        typer.Option("--formats", "-f", help="Comma-separated formats (jsonl,markdown,parquet)"),
    ] = "jsonl,markdown,parquet",
) -> None:
    """Export incidents to multiple formats."""
    console.print("[bold]Exporting Incidents[/bold]")
    incidents = _load_incidents_from_path(input_path)
    console.print(f"Loaded {len(incidents)} incidents")

    fmt_list = [f.strip().lower() for f in formats.split(",") if f.strip()]
    paths = asyncio.run(
        export_incidents(
            incidents,
            output_dir,
            export_markdown="markdown" in fmt_list,
            export_jsonl="jsonl" in fmt_list,
            export_parquet="parquet" in fmt_list,
        )
    )

    console.print("\n[green]Export complete![/green]")
    for fmt, path in paths.items():
        console.print(f"  {fmt}: {path}")


@app.command()
def report(
    input_path: Annotated[
        Path,
        typer.Argument(help="Directory or JSONL file containing incident JSONL files"),
    ] = Path("output"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for reports"),
    ] = Path("reports"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Report format (markdown, json, both)"),
    ] = "both",
    topology: Annotated[
        Path | None,
        typer.Option("--topology", "-t", help="Optional topology YAML for full semantic checks"),
    ] = None,
) -> None:
    """Generate quality report and dataset card."""
    console.print("[bold]Generating Quality Report[/bold]")
    incidents = _load_incidents_from_path(input_path)
    console.print(f"Loaded {len(incidents)} incidents")

    topology_obj = TopologyLoader().load(topology) if topology and topology.exists() else None
    paths = asyncio.run(
        generate_report(incidents, output_dir, format=format, topology=topology_obj)
    )

    console.print("\n[green]Report generated![/green]")
    for fmt, path in paths.items():
        console.print(f"  {fmt}: {path}")


@app.command()
def judge(
    input_path: Annotated[
        Path,
        typer.Argument(help="Directory or JSONL file containing incidents to evaluate"),
    ] = Path("output"),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON report path"),
    ] = None,
    sample: Annotated[
        int,
        typer.Option("--sample", "-n", help="Number of incidents to evaluate"),
    ] = 10,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Judge LLM model name"),
    ] = "gpt-4o-mini",
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="LLM API base URL"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="LLM API key"),
    ] = None,
) -> None:
    """Evaluate incident quality using LLM-as-judge rubric."""
    incidents = _load_incidents_from_path(input_path)[: max(1, sample)]
    llm_config = LLMConfig(
        model=model,
        base_url=base_url or "https://api.openai.com/v1",
        api_key=api_key,
        temperature=0.0,
    )

    async def _evaluate() -> list[dict[str, Any]]:
        client = create_llm_client(llm_config)
        scores: list[dict[str, Any]] = []
        try:
            for inc in incidents:
                prompt = render_judge_prompt(inc.model_dump())
                response = await client.generate(prompt=prompt)
                parsed = TolerantJSONParser.extract_json(response.content) or {}
                scores.append({"id": inc.id, "evaluation": parsed})
        finally:
            await client.close()
        return scores

    try:
        results = asyncio.run(_evaluate())
    except Exception as e:
        console.print(f"[red]Judge evaluation failed: {e}[/red]")
        raise typer.Exit(1) from e

    overalls = [
        float(r["evaluation"]["overall"])
        for r in results
        if isinstance(r.get("evaluation"), dict) and "overall" in r["evaluation"]
    ]
    avg_overall = sum(overalls) / len(overalls) if overalls else 0.0
    console.print(f"Evaluated {len(results)} incidents (avg overall: {avg_overall:.2f}/5)")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"average_overall": avg_overall, "results": results}, indent=2),
            encoding="utf-8",
        )
        console.print(f"[green]Judge report saved to: {output}[/green]")


@app.command()
def schema(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file for JSON schema"),
    ] = None,
) -> None:
    """Export JSON schema for Incident model."""
    schema_json = Incident.model_json_schema()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(schema_json, indent=2), encoding="utf-8")
        console.print(f"[green]Schema saved to: {output}[/green]")
    else:
        console.print(json.dumps(schema_json, indent=2))


if __name__ == "__main__":
    app()
