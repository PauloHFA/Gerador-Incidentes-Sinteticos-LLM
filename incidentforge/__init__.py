"""IncidentForge - Synthetic post-mortem and chaos scenario generator."""

__version__ = "0.1.0"
__author__ = "IncidentForge Contributors"
__license__ = "MIT"

from .client import LLMClient, LLMConfig, create_llm_client
from .dedup import DedupConfig, DiversityController
from .exporter import ExporterManager, export_incidents
from .failures import FailureCatalogLoader
from .generator import GeneratorConfig, IncidentGenerator
from .models import (
    ActionItem,
    ArchitectureProfile,
    FailureCatalog,
    FailureCategory,
    FailureTemplate,
    GenerationPlan,
    GenerationTask,
    Incident,
    IncidentMetadata,
    Language,
    Priority,
    Service,
    ServiceDependency,
    ServiceSLO,
    ServiceTechnology,
    Severity,
    SLAImpact,
    Style,
    TimelineEvent,
    Topology,
)
from .planner import PlannerConfig, TaskPlanner
from .report import QualityReporter, generate_report
from .topology import TopologyLoader
from .validators import (
    MultiLayerValidator,
    SchemaValidator,
    SecurityValidator,
    SemanticValidator,
    ValidationContext,
    ValidationResult,
    create_validator,
    validate_incident,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Incident",
    "Topology",
    "FailureCatalog",
    "Service",
    "ServiceDependency",
    "ServiceTechnology",
    "ServiceSLO",
    "FailureCategory",
    "FailureTemplate",
    "TimelineEvent",
    "ActionItem",
    "Severity",
    "Priority",
    "ArchitectureProfile",
    "Language",
    "Style",
    "SLAImpact",
    "IncidentMetadata",
    "GenerationPlan",
    "GenerationTask",
    "TopologyLoader",
    "FailureCatalogLoader",
    "TaskPlanner",
    "PlannerConfig",
    "IncidentGenerator",
    "GeneratorConfig",
    "LLMClient",
    "LLMConfig",
    "create_llm_client",
    "SchemaValidator",
    "SemanticValidator",
    "SecurityValidator",
    "MultiLayerValidator",
    "ValidationContext",
    "ValidationResult",
    "create_validator",
    "validate_incident",
    "DiversityController",
    "DedupConfig",
    "ExporterManager",
    "export_incidents",
    "QualityReporter",
    "generate_report",
]
