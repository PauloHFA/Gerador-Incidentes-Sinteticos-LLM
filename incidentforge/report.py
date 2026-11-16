"""Quality reporting and dataset card generation for IncidentForge."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dedup import DedupConfig, DiversityController
from .models import Incident, Topology
from .validators import MultiLayerValidator, SchemaValidator, SecurityValidator, ValidationContext

_PT_HINTS = (" de ", " da ", " do ", " em ", " para ", " com ", " não ", " falha ", " serviço ")


def _infer_language(incident: Incident) -> str:
    """Infer incident language from metadata or text content."""
    if incident.metadata.language is not None:
        return incident.metadata.language.value
    sample = f" {incident.root_cause_analysis.lower()} "
    hits = sum(1 for hint in _PT_HINTS if hint in sample)
    return "pt-BR" if hits >= 2 else "en"


class QualityReporter:
    """Generates quality reports for incident datasets."""

    def __init__(
        self,
        incidents: list[Incident],
        topology: Topology | None = None,
        rejected_count: int = 0,
        first_pass_count: int | None = None,
        use_embeddings: bool = False,
    ):
        self.incidents = incidents
        self.topology = topology
        self.rejected_count = rejected_count
        self.first_pass_count = first_pass_count
        self.controller = (
            DiversityController.from_incidents(
                incidents,
                config=DedupConfig(use_embeddings=use_embeddings, use_minhash=True),
            )
            if incidents
            else None
        )

    @classmethod
    def from_incidents(
        cls,
        incidents: list[Incident],
        topology: Topology | None = None,
        rejected_count: int = 0,
        first_pass_count: int | None = None,
        use_embeddings: bool = False,
    ) -> QualityReporter:
        """Create reporter from incidents."""
        return cls(
            incidents=incidents,
            topology=topology,
            rejected_count=rejected_count,
            first_pass_count=first_pass_count,
            use_embeddings=use_embeddings,
        )

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive quality report."""
        if not self.incidents:
            return {"error": "No incidents to report on"}

        return {
            "summary": self._generate_summary(),
            "distributions": self._generate_distributions(),
            "quality_metrics": self._generate_quality_metrics(),
            "diversity_metrics": self._generate_diversity_metrics(),
            "validation_metrics": self._generate_validation_metrics(),
        }

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        total = len(self.incidents)
        durations = [i.outage_duration_minutes for i in self.incidents]
        timeline_lengths = [len(i.timeline) for i in self.incidents]
        action_counts = [len(i.action_items) for i in self.incidents]
        root_cause_lengths = [len(i.root_cause_analysis) for i in self.incidents]

        return {
            "total_incidents": total,
            "unique_services": len({i.service for i in self.incidents}),
            "unique_categories": len({i.failure_category for i in self.incidents}),
            "date_range": {
                "earliest": min(i.metadata.generated_at for i in self.incidents),
                "latest": max(i.metadata.generated_at for i in self.incidents),
            },
            "duration_stats": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                "min": min(durations),
                "max": max(durations),
            },
            "timeline_stats": {
                "mean": statistics.mean(timeline_lengths),
                "median": statistics.median(timeline_lengths),
                "min": min(timeline_lengths),
                "max": max(timeline_lengths),
            },
            "action_items_stats": {
                "mean": statistics.mean(action_counts),
                "median": statistics.median(action_counts),
                "min": min(action_counts),
                "max": max(action_counts),
            },
            "root_cause_length_stats": {
                "mean": statistics.mean(root_cause_lengths),
                "median": statistics.median(root_cause_lengths),
                "min": min(root_cause_lengths),
                "max": max(root_cause_lengths),
            },
        }

    def _generate_distributions(self) -> dict[str, Any]:
        """Generate distribution statistics."""
        severity_dist = Counter(i.severity.value for i in self.incidents)
        category_dist = Counter(i.failure_category.value for i in self.incidents)
        service_dist = Counter(i.service for i in self.incidents)
        language_dist = Counter(_infer_language(i) for i in self.incidents)
        style_dist: Counter[str] = Counter()

        for i in self.incidents:
            if i.metadata.style is not None:
                style_dist[i.metadata.style.value] += 1

        return {
            "by_severity": dict(severity_dist),
            "by_category": dict(category_dist),
            "by_service": dict(service_dist.most_common(20)),
            "by_language": dict(language_dist),
            "by_style": dict(style_dist),
        }

    def _generate_quality_metrics(self) -> dict[str, Any]:
        """Generate quality metrics."""
        total_seen = len(self.incidents) + self.rejected_count
        rejection_rate = self.rejected_count / total_seen if total_seen > 0 else 0.0

        if self.first_pass_count is not None and total_seen > 0:
            first_pass_rate = self.first_pass_count / total_seen
        else:
            first_pass_rate = 0.0

        schema_valid = sum(1 for i in self.incidents if self._validate_schema(i))
        schema_rate = schema_valid / len(self.incidents) if self.incidents else 0.0

        return {
            "rejection_rate": rejection_rate,
            "first_pass_validation_rate": first_pass_rate,
            "schema_validation_rate": schema_rate,
            "total_tokens_estimate": sum(
                (i.metadata.tokens_input or 0) + (i.metadata.tokens_output or 0)
                for i in self.incidents
            ),
        }

    def _generate_diversity_metrics(self) -> dict[str, Any]:
        """Generate diversity metrics."""
        if not self.controller:
            return {}

        stats = self.controller.get_stats()

        all_text = " ".join(
            i.root_cause_analysis + " " + " ".join(e.event for e in i.timeline)
            for i in self.incidents
        )
        words = all_text.split()
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0.0

        cat_dist = stats.get("category_distribution", {})
        if cat_dist and len(cat_dist) > 1:
            probs = list(cat_dist.values())
            entropy = float(-sum(p * np.log2(p) for p in probs if p > 0))
            max_entropy = float(np.log2(len(probs)))
            balance = entropy / max_entropy if max_entropy > 0 else 0.0
        elif cat_dist:
            balance = 1.0
        else:
            balance = 0.0

        return {
            "lexical_diversity": lexical_diversity,
            "category_balance": balance,
            "category_distribution": cat_dist,
            "deduplication_stats": {
                "embedding_index_size": stats.get("embedding_index_size", 0),
                "minhash_index_size": stats.get("minhash_index_size", 0),
            },
        }

    def _generate_validation_metrics(self) -> dict[str, Any]:
        """Run actual validation layers across incidents."""
        schema_valid = 0
        semantic_valid = 0
        security_valid = 0

        for incident in self.incidents:
            if self._validate_schema(incident):
                schema_valid += 1

            service = self.topology.get_service(incident.service) if self.topology else None
            if self.topology is not None and service is not None:
                results = MultiLayerValidator(
                    ValidationContext(topology=self.topology, service=service)
                ).validate(incident)
                for r in results:
                    if r.layer == "semantic" and r.valid:
                        semantic_valid += 1
                    elif r.layer == "security" and r.valid:
                        security_valid += 1
            else:
                sec = SecurityValidator(
                    ValidationContext(
                        topology=self.topology
                        if self.topology is not None
                        else Topology.model_construct(version="1.0", services=[]),
                        service=service if service is not None else None,  # type: ignore[arg-type]
                    )
                ).validate(incident)
                if sec.valid:
                    security_valid += 1

        return {
            "schema_valid": schema_valid,
            "semantic_valid": semantic_valid,
            "security_valid": security_valid,
            "total_validated": len(self.incidents),
        }

    def _validate_schema(self, incident: Incident) -> bool:
        """Quick schema validation."""
        return SchemaValidator.validate(incident).valid

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        report = self.generate_report()
        if "error" in report:
            return f"# IncidentForge Dataset Quality Report\n\n{report['error']}\n"

        generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        lines = [
            "# IncidentForge Dataset Quality Report",
            "",
            f"Generated: {generated_at}",
            "",
            "## Summary",
            "",
            f"- **Total Incidents:** {report['summary']['total_incidents']}",
            f"- **Unique Services:** {report['summary']['unique_services']}",
            f"- **Unique Categories:** {report['summary']['unique_categories']}",
            f"- **Date Range:** {report['summary']['date_range']['earliest']} to {report['summary']['date_range']['latest']}",
            "",
            "### Duration Statistics (minutes)",
            "",
            f"- **Mean:** {report['summary']['duration_stats']['mean']:.1f}",
            f"- **Median:** {report['summary']['duration_stats']['median']:.1f}",
            f"- **Std Dev:** {report['summary']['duration_stats']['stdev']:.1f}",
            f"- **Range:** {report['summary']['duration_stats']['min']} - {report['summary']['duration_stats']['max']}",
            "",
            "### Timeline Statistics",
            "",
            f"- **Mean Events:** {report['summary']['timeline_stats']['mean']:.1f}",
            f"- **Median Events:** {report['summary']['timeline_stats']['median']:.1f}",
            f"- **Range:** {report['summary']['timeline_stats']['min']} - {report['summary']['timeline_stats']['max']}",
            "",
            "### Action Items Statistics",
            "",
            f"- **Mean:** {report['summary']['action_items_stats']['mean']:.1f}",
            f"- **Median:** {report['summary']['action_items_stats']['median']:.1f}",
            f"- **Range:** {report['summary']['action_items_stats']['min']} - {report['summary']['action_items_stats']['max']}",
            "",
            "### Root Cause Length Statistics",
            "",
            f"- **Mean:** {report['summary']['root_cause_length_stats']['mean']:.0f} chars",
            f"- **Median:** {report['summary']['root_cause_length_stats']['median']:.0f} chars",
            f"- **Range:** {report['summary']['root_cause_length_stats']['min']} - {report['summary']['root_cause_length_stats']['max']}",
            "",
            "## Distributions",
            "",
            "### By Severity",
            "",
        ]

        total = report["summary"]["total_incidents"]
        for sev, count in sorted(report["distributions"]["by_severity"].items()):
            pct = count / total * 100
            lines.append(f"- **{sev}:** {count} ({pct:.1f}%)")

        lines.extend(["", "### By Failure Category (Top 15)", ""])
        for cat, count in sorted(
            report["distributions"]["by_category"].items(), key=lambda x: -x[1]
        )[:15]:
            pct = count / total * 100
            lines.append(f"- **{cat}:** {count} ({pct:.1f}%)")

        lines.extend(["", "### By Service (Top 15)", ""])
        for svc, count in sorted(
            report["distributions"]["by_service"].items(), key=lambda x: -x[1]
        )[:15]:
            pct = count / total * 100
            lines.append(f"- **{svc}:** {count} ({pct:.1f}%)")

        lines.extend(["", "### By Language", ""])
        for lang, count in sorted(report["distributions"]["by_language"].items()):
            pct = count / total * 100
            lines.append(f"- **{lang}:** {count} ({pct:.1f}%)")

        lines.extend(
            [
                "",
                "## Quality Metrics",
                "",
                f"- **Schema Validation Rate:** {report['quality_metrics']['schema_validation_rate']:.1%}",
                f"- **First-Pass Validation Rate:** {report['quality_metrics']['first_pass_validation_rate']:.1%}",
                f"- **Rejection Rate:** {report['quality_metrics']['rejection_rate']:.1%}",
                f"- **Estimated Total Tokens:** {report['quality_metrics']['total_tokens_estimate']:,}",
                "",
                "## Diversity Metrics",
                "",
                f"- **Lexical Diversity:** {report['diversity_metrics']['lexical_diversity']:.3f}",
                f"- **Category Balance (Entropy):** {report['diversity_metrics']['category_balance']:.3f}",
                "",
                "### Category Distribution",
                "",
            ]
        )

        cat_items = sorted(
            report["diversity_metrics"]["category_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for cat, pct in cat_items:
            lines.append(f"- **{cat}:** {pct:.1%}")

        lines.extend(["", "---", "", "*Report generated by IncidentForge*"])
        return "\n".join(lines)

    def to_json(self) -> str:
        """Generate JSON report."""
        return json.dumps(self.generate_report(), indent=2, default=str, ensure_ascii=False)


class DatasetCardGenerator:
    """Generates Hugging Face dataset card."""

    def __init__(
        self,
        incidents: list[Incident],
        model_name: str = "incidentforge",
        license: str = "MIT",
        use_embeddings: bool = False,
    ):
        self.incidents = incidents
        self.model_name = model_name
        self.license = license
        self.reporter = QualityReporter.from_incidents(incidents, use_embeddings=use_embeddings)

    def generate_card(self) -> str:
        """Generate dataset card in Markdown."""
        report = self.reporter.generate_report()
        if "error" in report:
            return f"# {self.model_name} Synthetic Incident Dataset\n\nNo incidents available.\n"

        summary = report["summary"]
        distributions = report["distributions"]
        diversity = report["diversity_metrics"]

        lines = [
            "---",
            f"license: {self.license.lower()}",
            "tags:",
            "  - synthetic-data",
            "  - incident-postmortem",
            "  - site-reliability-engineering",
            "  - chaos-engineering",
            "  - observability",
            "  - rag",
            "  - llm-training",
            "language:",
            "  - en",
            "  - pt",
            "task_categories:",
            "  - text-generation",
            "  - conditional-text-generation",
            "pretty_name: IncidentForge Synthetic Incidents",
            "---",
            "",
            f"# {self.model_name} Synthetic Incident Dataset",
            "",
            "## Dataset Description",
            "",
            "This dataset contains synthetically generated production incident post-mortems "
            "created by [IncidentForge](https://github.com/PauloHFA/IncidentForge). "
            "The incidents are generated from system topology descriptions using LLMs, "
            "and are designed for:",
            "",
            "- Training and evaluating RAG systems for incident response",
            "- Testing observability platforms and alerting systems",
            "- Incident response team training and tabletop exercises",
            "- Chaos engineering scenario planning",
            "- Fine-tuning LLMs for technical incident analysis",
            "",
            "**Important:** All incidents in this dataset are 100% synthetic and do not represent "
            "real production incidents. They should not be used as factual references for "
            "real system failures.",
            "",
            "## Dataset Statistics",
            "",
            f"- **Total Incidents:** {summary['total_incidents']:,}",
            f"- **Unique Services:** {summary['unique_services']}",
            f"- **Unique Failure Categories:** {summary['unique_categories']}",
            "- **Languages:** English, Portuguese (Brazil)",
            "- **Styles:** Blameless Formal, Terse, Detailed",
            "- **Architecture Profiles:** Microservices, Monolith Legacy, Serverless",
            "",
            "### Severity Distribution",
            "",
        ]

        for sev, count in sorted(distributions["by_severity"].items()):
            pct = count / summary["total_incidents"] * 100
            lines.append(f"- **{sev}:** {count} ({pct:.1f}%)")

        lines.extend(["", "### Failure Category Distribution (Top 10)", ""])
        for cat, count in sorted(distributions["by_category"].items(), key=lambda x: -x[1])[:10]:
            pct = count / summary["total_incidents"] * 100
            lines.append(f"- **{cat}:** {count} ({pct:.1f}%)")

        lines.extend(["", "### Service Distribution (Top 10)", ""])
        for svc, count in sorted(distributions["by_service"].items(), key=lambda x: -x[1])[:10]:
            pct = count / summary["total_incidents"] * 100
            lines.append(f"- **{svc}:** {count} ({pct:.1f}%)")

        lines.extend(
            [
                "",
                "## Data Fields",
                "",
                "| Field | Type | Description |",
                "|-------|------|-------------|",
                "| `id` | string | Unique incident ID (INC-YYYY-NNNN) |",
                "| `service` | string | Affected service name |",
                "| `severity` | string | SEV-1, SEV-2, SEV-3, or SEV-4 |",
                "| `outage_duration_minutes` | integer | Outage duration in minutes |",
                "| `failure_category` | string | Failure category from catalog |",
                "| `sla_impact` | object | Availability before/after, P99 latency |",
                "| `root_cause_analysis` | string | Root cause analysis (min 150 chars) |",
                "| `timeline` | array | Chronological events (min 5) |",
                "| `action_items` | array | 2-8 action items with owner/priority |",
                "| `metadata` | object | Generation metadata (model, seed, etc.) |",
                "",
                "## Data Splits",
                "",
                "The dataset is provided as a single split (`train`) containing all incidents. "
                "Users can create their own splits based on severity, category, service, or other criteria.",
                "",
                "## Generation Method",
                "",
                "Incidents are generated using a multi-stage pipeline:",
                "",
                "1. **Topology Definition**: System topology described in YAML (services, technologies, dependencies, SLOs)",
                "2. **Failure Catalog**: 30+ failure categories across database, network, messaging, compute, cache, deploy",
                "3. **Task Planning**: Deterministic cartesian product of services x failures x severities with seed-based reproducibility",
                "4. **LLM Generation**: Structured output via JSON schema with provider-agnostic client (OpenAI, vLLM, Ollama, NIM)",
                "5. **Multi-Layer Validation**: Schema (Pydantic), Semantic (chronology, topology consistency), Security (PII/secrets)",
                "6. **Deduplication**: Semantic similarity (embeddings + MinHash) with category balance enforcement",
                "7. **Export**: Markdown, JSONL, and Parquet formats",
                "",
                "## Quality Metrics",
                "",
                f"- **Lexical Diversity:** {diversity['lexical_diversity']:.3f}",
                f"- **Category Balance (Entropy):** {diversity['category_balance']:.3f}",
                f"- **Schema Validation Rate:** {report['quality_metrics']['schema_validation_rate']:.1%}",
                "",
                "## Limitations",
                "",
                "- **Synthetic Data**: All incidents are generated by LLMs and may not reflect real-world complexity",
                "- **Model Bias**: Quality depends on the LLM used for generation",
                "- **Topology Dependency**: Incidents are only as realistic as the input topology",
                "- **No Real Logs**: Incidents do not contain actual log lines, metrics, or traces",
                "- **Language Coverage**: Primarily English and Portuguese (Brazil)",
                "",
                "## Citation",
                "",
                "```bibtex",
                "@misc{incidentforge,",
                "  title={IncidentForge: Synthetic Incident Post-Mortem Generator},",
                "  author={IncidentForge Contributors},",
                "  year={2026},",
                "  url={https://github.com/PauloHFA/IncidentForge}",
                "}",
                "```",
                "",
                "## License",
                "",
                f"This dataset is released under the {self.license} License. "
                "The synthetic incidents are generated using LLMs whose outputs may be subject to "
                "the respective model's terms of use. Please review the model's license before "
                "using this dataset commercially.",
                "",
                "## Contact",
                "",
                "For questions or contributions, visit [IncidentForge GitHub](https://github.com/PauloHFA/IncidentForge).",
            ]
        )

        return "\n".join(lines)

    def save(self, output_path: Path) -> Path:
        """Save dataset card to file."""
        content = self.generate_card()
        output_path.write_text(content, encoding="utf-8")
        return output_path


async def generate_report(
    incidents: list[Incident],
    output_dir: Path,
    format: str = "both",
    topology: Topology | None = None,
    rejected_count: int = 0,
    first_pass_count: int | None = None,
    use_embeddings: bool = False,
) -> dict[str, Path]:
    """Generate quality report and dataset card."""
    output_dir.mkdir(parents=True, exist_ok=True)

    reporter = QualityReporter.from_incidents(
        incidents,
        topology=topology,
        rejected_count=rejected_count,
        first_pass_count=first_pass_count,
        use_embeddings=use_embeddings,
    )
    card_generator = DatasetCardGenerator(incidents, use_embeddings=use_embeddings)

    paths: dict[str, Path] = {}

    if format in ("markdown", "both"):
        md_path = output_dir / "quality_report.md"
        md_path.write_text(reporter.to_markdown(), encoding="utf-8")
        paths["markdown"] = md_path

    if format in ("json", "both"):
        json_path = output_dir / "quality_report.json"
        json_path.write_text(reporter.to_json(), encoding="utf-8")
        paths["json"] = json_path

    card_path = output_dir / "DATASET_CARD.md"
    card_generator.save(card_path)
    paths["dataset_card"] = card_path

    return paths
