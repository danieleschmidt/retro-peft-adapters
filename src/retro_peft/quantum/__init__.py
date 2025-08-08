"""
Quantum-Classical Hybrid Computing for Retro-PEFT-Adapters

This module implements quantum-classical hybrid architectures and prepares
the framework for integration with quantum computing systems, including:
- Quantum-inspired classical algorithms
- Quantum simulation capabilities
- Hybrid quantum-classical neural networks
- Quantum advantage detection and routing
- Future quantum hardware integration interfaces
"""

from .hybrid_training import HybridTrainingEngine, QuantumAdvantageDetector
from .quantum_adapters import HybridQuantumClassicalAdapter, QuantumInspiredAdapter
from .quantum_algorithms import QuantumFeatureMap, QuantumInspiredOptimizer
from .quantum_simulation import QuantumCircuitBuilder, QuantumSimulator

__all__ = [
    "QuantumInspiredAdapter",
    "HybridQuantumClassicalAdapter",
    "QuantumInspiredOptimizer",
    "QuantumFeatureMap",
    "QuantumSimulator",
    "QuantumCircuitBuilder",
    "HybridTrainingEngine",
    "QuantumAdvantageDetector",
]
