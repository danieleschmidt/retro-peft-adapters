"""
Federated Learning for Retro-PEFT-Adapters

This module implements federated learning capabilities for distributed training
of retrieval-augmented adapters across multiple organizations while preserving
data privacy and security.
"""

from .client import FederatedClient
from .server import FederatedServer
from .aggregation import FederatedAggregator
from .privacy import PrivacyEngine
from .communication import SecureCommunicator

__all__ = [
    "FederatedClient",
    "FederatedServer", 
    "FederatedAggregator",
    "PrivacyEngine",
    "SecureCommunicator"
]