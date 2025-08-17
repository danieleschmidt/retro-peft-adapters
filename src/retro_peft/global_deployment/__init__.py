"""
Global-First Production Deployment for Retro-PEFT-Adapters

Comprehensive global deployment system with multi-region orchestration,
intelligent scaling, edge computing integration, and compliance automation
for worldwide production deployment of retrieval-augmented adapters.

Key Components:
- GlobalOrchestrator: Multi-region deployment coordination
- EdgeDeployment: Edge computing and CDN integration  
- ComplianceManager: Automated regulatory compliance (GDPR, CCPA, etc.)
- IntelligentScaling: Adaptive resource management across regions
- MultiCloudManager: Cross-cloud deployment and orchestration
- I18nManager: Internationalization and localization automation
"""

from .global_orchestrator import (
    GlobalOrchestrator,
    DeploymentRegion,
    GlobalConfig,
    RegionHealth
)

from .edge_deployment import (
    EdgeDeployment,
    EdgeNode,
    CDNIntegration,
    EdgeOptimizer
)

from .compliance_manager import (
    ComplianceManager,
    GDPRCompliance,
    CCPACompliance,
    DataSovereignty
)

from .intelligent_scaling import (
    IntelligentScaling,
    AutoScaler,
    ResourcePredictor,
    LoadBalancer
)

from .multi_cloud_manager import (
    MultiCloudManager,
    CloudProvider,
    CrossCloudSync,
    FailoverManager
)

from .i18n_manager import (
    I18nManager,
    LocalizationEngine,
    CultureAdapter,
    TranslationService
)

__all__ = [
    "GlobalOrchestrator",
    "DeploymentRegion",
    "GlobalConfig", 
    "RegionHealth",
    "EdgeDeployment",
    "EdgeNode",
    "CDNIntegration",
    "EdgeOptimizer",
    "ComplianceManager",
    "GDPRCompliance",
    "CCPACompliance",
    "DataSovereignty",
    "IntelligentScaling",
    "AutoScaler",
    "ResourcePredictor", 
    "LoadBalancer",
    "MultiCloudManager",
    "CloudProvider",
    "CrossCloudSync",
    "FailoverManager",
    "I18nManager",
    "LocalizationEngine",
    "CultureAdapter",
    "TranslationService"
]