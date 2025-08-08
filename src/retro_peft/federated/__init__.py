"""
Federated Learning for Retro-PEFT-Adapters

This module implements federated learning capabilities for distributed training
of retrieval-augmented adapters across multiple organizations while preserving
data privacy and security.
"""

from .aggregation import FederatedAggregator
from .client import FederatedClient
from .communication import SecureCommunicator
from .privacy import PrivacyEngine
from .server import FederatedServer

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "FederatedAggregator",
    "PrivacyEngine",
    "SecureCommunicator",
]
