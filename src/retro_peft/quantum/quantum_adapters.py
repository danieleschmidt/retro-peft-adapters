"""
Quantum-Inspired and Hybrid Quantum-Classical Adapters

This module implements quantum-inspired adapter architectures that leverage
quantum computing principles while remaining executable on classical hardware.
Includes preparation for future quantum hardware integration.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm

from ..adapters.base_adapter import BaseRetroAdapter

logger = logging.getLogger(__name__)


@dataclass
class QuantumConfig:
    """Configuration for quantum-inspired adapters"""

    n_qubits: int = 8
    n_layers: int = 4
    rotation_gates: List[str] = None
    entanglement_pattern: str = "linear"  # linear, circular, full
    measurement_shots: int = 1024
    noise_model: Optional[str] = None
    use_parameter_sharing: bool = True
    quantum_advantage_threshold: float = 0.1


class QuantumInspiredAdapter(nn.Module):
    """
    Quantum-inspired adapter using quantum computing principles

    This adapter implements quantum-inspired algorithms that can provide
    computational advantages for certain types of optimization problems
    commonly found in neural network adaptation.

    Features:
    - Quantum-inspired parameterization
    - Variational quantum circuits simulation
    - Quantum feature maps
    - Entanglement-aware parameter updates
    """

    def __init__(self, input_dim: int, output_dim: int, config: QuantumConfig):
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = config.n_qubits

        # Ensure we have enough qubits for the problem
        min_qubits = math.ceil(math.log2(max(input_dim, output_dim)))
        if self.n_qubits < min_qubits:
            self.n_qubits = min_qubits
            logger.warning(f"Increased qubits to {self.n_qubits} to accommodate dimensions")

        # Quantum-inspired parameter layers
        self.quantum_feature_map = QuantumFeatureMap(input_dim, self.n_qubits)
        self.variational_circuit = VariationalQuantumCircuit(
            n_qubits=self.n_qubits,
            n_layers=config.n_layers,
            rotation_gates=config.rotation_gates or ["rx", "ry", "rz"],
            entanglement_pattern=config.entanglement_pattern,
        )

        # Classical post-processing layers
        self.classical_projection = nn.Sequential(
            nn.Linear(2**self.n_qubits if self.n_qubits <= 10 else 2**10, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Quantum measurement simulation
        self.measurement_basis = nn.Parameter(torch.randn(self.n_qubits, 2, 2))

        # Parameter sharing for quantum efficiency
        if config.use_parameter_sharing:
            self._setup_parameter_sharing()

        self.dropout = nn.Dropout(0.1)

    def _setup_parameter_sharing(self):
        """Setup parameter sharing to reduce quantum circuit complexity"""
        # Share parameters across similar gate operations
        shared_params = {}
        for name, param in self.variational_circuit.named_parameters():
            if "rotation" in name:
                gate_type = name.split("_")[0]
                if gate_type not in shared_params:
                    shared_params[gate_type] = param
                else:
                    # Replace with shared parameter
                    param.data = shared_params[gate_type].data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum-inspired adapter

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Adapted output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Quantum feature encoding
        quantum_features = self.quantum_feature_map(x)

        # Variational quantum circuit processing
        quantum_state = self.variational_circuit(quantum_features)

        # Quantum measurement simulation
        measurement_probs = self._simulate_quantum_measurement(quantum_state)

        # Classical post-processing
        output = self.classical_projection(measurement_probs)
        output = self.dropout(output)

        return output

    def _simulate_quantum_measurement(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Simulate quantum measurement to extract classical information

        Args:
            quantum_state: Quantum state representation

        Returns:
            Measurement probability distribution
        """
        batch_size, seq_len, state_dim = quantum_state.shape

        # Apply measurement operators
        measurements = []
        for qubit_idx in range(self.n_qubits):
            # Pauli-Z measurement on each qubit
            measurement_op = self.measurement_basis[qubit_idx]

            # Project quantum state onto measurement basis
            measured_state = torch.matmul(quantum_state, measurement_op)
            measurement_prob = torch.sum(measured_state**2, dim=-1, keepdim=True)
            measurements.append(measurement_prob)

        # Combine measurements into probability distribution
        measurement_probs = torch.cat(measurements, dim=-1)

        # Normalize probabilities
        measurement_probs = F.softmax(measurement_probs, dim=-1)

        return measurement_probs


class HybridQuantumClassicalAdapter(nn.Module):
    """
    Hybrid quantum-classical adapter with adaptive routing

    This adapter dynamically routes computations between quantum-inspired
    and classical pathways based on the detected potential for quantum advantage.
    """

    def __init__(self, input_dim: int, output_dim: int, config: QuantumConfig):
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Quantum pathway
        self.quantum_adapter = QuantumInspiredAdapter(input_dim, output_dim, config)

        # Classical pathway (standard LoRA-style)
        self.classical_adapter = ClassicalAdapter(input_dim, output_dim)

        # Quantum advantage detector
        self.advantage_detector = QuantumAdvantageDetector(
            input_dim=input_dim, threshold=config.quantum_advantage_threshold
        )

        # Adaptive gating mechanism
        self.routing_gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Binary choice: quantum vs classical
            nn.Softmax(dim=-1),
        )

        # Performance tracking
        self.quantum_usage_count = 0
        self.classical_usage_count = 0
        self.performance_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive quantum-classical routing

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Adapted output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, input_dim = x.shape

        # Detect potential quantum advantage
        quantum_advantage_score = self.advantage_detector(x)

        # Compute routing probabilities
        x_pooled = x.mean(dim=1)  # Global average pooling
        routing_probs = self.routing_gate(x_pooled)

        # Weighted routing based on quantum advantage and learned preferences
        quantum_weight = (routing_probs[:, 0] * quantum_advantage_score).unsqueeze(-1).unsqueeze(-1)
        classical_weight = (
            (routing_probs[:, 1] * (1.0 - quantum_advantage_score)).unsqueeze(-1).unsqueeze(-1)
        )

        # Normalize weights
        total_weight = quantum_weight + classical_weight + 1e-8
        quantum_weight = quantum_weight / total_weight
        classical_weight = classical_weight / total_weight

        # Process through both pathways
        quantum_output = self.quantum_adapter(x)
        classical_output = self.classical_adapter(x)

        # Weighted combination
        hybrid_output = quantum_weight * quantum_output + classical_weight * classical_output

        # Update usage statistics
        avg_quantum_weight = quantum_weight.mean().item()
        if avg_quantum_weight > 0.5:
            self.quantum_usage_count += 1
        else:
            self.classical_usage_count += 1

        # Track performance for adaptive learning
        self.performance_history.append(
            {
                "quantum_weight": avg_quantum_weight,
                "quantum_advantage_score": quantum_advantage_score.mean().item(),
                "batch_size": batch_size,
            }
        )

        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

        return hybrid_output

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for quantum vs classical pathways"""
        total_usage = self.quantum_usage_count + self.classical_usage_count

        if total_usage == 0:
            return {
                "quantum_usage_ratio": 0.0,
                "classical_usage_ratio": 0.0,
                "total_forward_passes": 0,
                "average_quantum_advantage": 0.0,
            }

        avg_quantum_advantage = 0.0
        if self.performance_history:
            avg_quantum_advantage = np.mean(
                [h["quantum_advantage_score"] for h in self.performance_history]
            )

        return {
            "quantum_usage_ratio": self.quantum_usage_count / total_usage,
            "classical_usage_ratio": self.classical_usage_count / total_usage,
            "total_forward_passes": total_usage,
            "average_quantum_advantage": avg_quantum_advantage,
            "quantum_usage_count": self.quantum_usage_count,
            "classical_usage_count": self.classical_usage_count,
        }


class QuantumFeatureMap(nn.Module):
    """
    Quantum feature map for encoding classical data into quantum states

    Implements various encoding strategies inspired by quantum machine learning
    including amplitude encoding, angle encoding, and basis encoding.
    """

    def __init__(self, input_dim: int, n_qubits: int, encoding_type: str = "angle"):
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type

        # Feature scaling and preprocessing
        self.feature_scaler = nn.Sequential(
            nn.Linear(input_dim, n_qubits * 2), nn.Tanh()  # Bound features to [-1, 1]
        )

        # Quantum encoding parameters
        if encoding_type == "angle":
            self.encoding_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)
        elif encoding_type == "amplitude":
            # For amplitude encoding, need 2^n amplitudes
            max_amplitudes = min(2**n_qubits, 512)  # Limit for computational tractability
            self.amplitude_projection = nn.Linear(input_dim, max_amplitudes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical features into quantum representation

        Args:
            x: Input features [batch_size, seq_len, input_dim]

        Returns:
            Quantum feature representation [batch_size, seq_len, 2^n_qubits]
        """
        batch_size, seq_len, _ = x.shape

        # Scale features
        scaled_features = self.feature_scaler(x)

        if self.encoding_type == "angle":
            return self._angle_encoding(scaled_features)
        elif self.encoding_type == "amplitude":
            return self._amplitude_encoding(scaled_features)
        else:
            return self._basis_encoding(scaled_features)

    def _angle_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features using rotation angles"""
        batch_size, seq_len, _ = features.shape

        # Reshape features for angle encoding
        angles = features.view(batch_size, seq_len, self.n_qubits, 2)

        # Create quantum state representation using rotation matrices
        quantum_states = []
        for qubit in range(self.n_qubits):
            # Extract rotation angles for this qubit
            theta = angles[:, :, qubit, 0] + self.encoding_params[qubit, 0]
            phi = angles[:, :, qubit, 1] + self.encoding_params[qubit, 1]

            # Create qubit state |psi> = cos(θ/2)|0> + e^(iφ)sin(θ/2)|1>
            cos_term = torch.cos(theta / 2)
            sin_term = torch.sin(theta / 2)

            # Real and imaginary parts (simplified for classical simulation)
            real_part = cos_term
            imag_part = sin_term * torch.cos(phi)

            qubit_state = torch.stack([real_part, imag_part], dim=-1)
            quantum_states.append(qubit_state)

        # Tensor product of all qubit states (simplified)
        quantum_state = self._tensor_product_states(quantum_states)

        return quantum_state

    def _amplitude_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features directly as quantum amplitudes"""
        batch_size, seq_len, _ = features.shape

        # Project to amplitude space
        amplitudes = self.amplitude_projection(features)

        # Normalize amplitudes (quantum states must be normalized)
        amplitudes = F.normalize(amplitudes, p=2, dim=-1)

        return amplitudes

    def _basis_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features using computational basis states"""
        batch_size, seq_len, feature_dim = features.shape

        # Convert features to binary representation
        # This is a simplified version - real implementation would be more sophisticated
        binary_features = torch.sigmoid(features) > 0.5

        # Create basis state representation
        basis_states = torch.zeros(batch_size, seq_len, 2**self.n_qubits, device=features.device)

        for i in range(min(feature_dim, self.n_qubits)):
            state_index = binary_features[:, :, i].long()
            basis_states.scatter_(-1, state_index.unsqueeze(-1), 1.0)

        return basis_states

    def _tensor_product_states(self, qubit_states: List[torch.Tensor]) -> torch.Tensor:
        """Compute tensor product of individual qubit states"""
        # Simplified tensor product for classical simulation
        # In practice, this would be more complex for full quantum simulation

        if not qubit_states:
            return torch.zeros(1, 1, 1)

        result = qubit_states[0]
        for state in qubit_states[1:]:
            # Simplified tensor product
            result = torch.cat([result, state], dim=-1)

        return result


class VariationalQuantumCircuit(nn.Module):
    """
    Variational quantum circuit implementation for quantum-inspired processing

    Simulates parameterized quantum circuits with trainable parameters
    that can be optimized through gradient descent.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        rotation_gates: List[str],
        entanglement_pattern: str = "linear",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation_gates = rotation_gates
        self.entanglement_pattern = entanglement_pattern

        # Parameterized rotation gates
        self.rotation_params = nn.ParameterDict()
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                for gate in rotation_gates:
                    param_name = f"layer_{layer}_qubit_{qubit}_{gate}"
                    self.rotation_params[param_name] = nn.Parameter(torch.randn(1) * 0.1)

        # Entanglement gate parameters (if needed)
        if entanglement_pattern != "none":
            self.entanglement_params = nn.ParameterDict()
            entanglement_pairs = self._get_entanglement_pairs()
            for layer in range(n_layers):
                for i, (qubit1, qubit2) in enumerate(entanglement_pairs):
                    param_name = f"layer_{layer}_ent_{i}"
                    self.entanglement_params[param_name] = nn.Parameter(torch.randn(1) * 0.1)

    def _get_entanglement_pairs(self) -> List[Tuple[int, int]]:
        """Get qubit pairs for entanglement gates based on pattern"""
        pairs = []

        if self.entanglement_pattern == "linear":
            pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
        elif self.entanglement_pattern == "circular":
            pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        elif self.entanglement_pattern == "full":
            pairs = [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]

        return pairs

    def forward(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Process quantum state through variational circuit

        Args:
            quantum_state: Input quantum state

        Returns:
            Processed quantum state
        """
        current_state = quantum_state

        # Apply variational layers
        for layer in range(self.n_layers):
            # Apply rotation gates
            current_state = self._apply_rotation_layer(current_state, layer)

            # Apply entanglement gates
            if self.entanglement_pattern != "none":
                current_state = self._apply_entanglement_layer(current_state, layer)

        return current_state

    def _apply_rotation_layer(self, state: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply parameterized rotation gates to all qubits"""
        batch_size, seq_len, state_dim = state.shape

        # For each qubit, apply rotation gates
        processed_state = state.clone()

        for qubit in range(self.n_qubits):
            for gate in self.rotation_gates:
                param_name = f"layer_{layer}_qubit_{qubit}_{gate}"
                angle = self.rotation_params[param_name]

                # Apply rotation (simplified classical simulation)
                if gate == "rx":
                    rotation_matrix = self._rx_matrix(angle)
                elif gate == "ry":
                    rotation_matrix = self._ry_matrix(angle)
                elif gate == "rz":
                    rotation_matrix = self._rz_matrix(angle)

                # Apply rotation to state (simplified)
                processed_state = self._apply_single_qubit_gate(
                    processed_state, rotation_matrix, qubit
                )

        return processed_state

    def _apply_entanglement_layer(self, state: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply entanglement gates between qubits"""
        # Simplified entanglement simulation
        # In practice, this would implement CNOT or other two-qubit gates

        entanglement_pairs = self._get_entanglement_pairs()
        processed_state = state

        for i, (qubit1, qubit2) in enumerate(entanglement_pairs):
            param_name = f"layer_{layer}_ent_{i}"
            if param_name in self.entanglement_params:
                entanglement_strength = torch.sigmoid(self.entanglement_params[param_name])

                # Apply simplified entanglement
                processed_state = self._apply_entanglement_gate(
                    processed_state, qubit1, qubit2, entanglement_strength
                )

        return processed_state

    def _rx_matrix(self, angle: torch.Tensor) -> torch.Tensor:
        """Create RX rotation matrix"""
        cos_val = torch.cos(angle / 2)
        sin_val = torch.sin(angle / 2)
        return torch.stack(
            [torch.stack([cos_val, -1j * sin_val]), torch.stack([-1j * sin_val, cos_val])]
        )

    def _ry_matrix(self, angle: torch.Tensor) -> torch.Tensor:
        """Create RY rotation matrix"""
        cos_val = torch.cos(angle / 2)
        sin_val = torch.sin(angle / 2)
        return torch.stack([torch.stack([cos_val, -sin_val]), torch.stack([sin_val, cos_val])])

    def _rz_matrix(self, angle: torch.Tensor) -> torch.Tensor:
        """Create RZ rotation matrix"""
        exp_pos = torch.exp(1j * angle / 2)
        exp_neg = torch.exp(-1j * angle / 2)
        return torch.stack(
            [
                torch.stack([exp_neg, torch.zeros_like(angle)]),
                torch.stack([torch.zeros_like(angle), exp_pos]),
            ]
        )

    def _apply_single_qubit_gate(
        self, state: torch.Tensor, gate_matrix: torch.Tensor, qubit_index: int
    ) -> torch.Tensor:
        """Apply single qubit gate to quantum state"""
        # Simplified application - in practice would need full tensor product
        # For now, apply a linear transformation inspired by the gate

        batch_size, seq_len, state_dim = state.shape

        # Extract real part of gate matrix for classical simulation
        gate_real = gate_matrix.real if torch.is_complex(gate_matrix) else gate_matrix

        # Apply transformation (simplified)
        transformation = torch.eye(state_dim, device=state.device)
        if qubit_index < state_dim:
            # Modify specific dimensions corresponding to the qubit
            start_idx = qubit_index * 2 % state_dim
            end_idx = min(start_idx + 2, state_dim)
            if end_idx - start_idx == 2:
                transformation[start_idx:end_idx, start_idx:end_idx] = gate_real[:2, :2]

        return torch.matmul(state, transformation)

    def _apply_entanglement_gate(
        self, state: torch.Tensor, qubit1: int, qubit2: int, strength: torch.Tensor
    ) -> torch.Tensor:
        """Apply entanglement gate between two qubits"""
        # Simplified entanglement - mix information between qubit representations
        batch_size, seq_len, state_dim = state.shape

        # Create mixing matrix
        mixing_matrix = torch.eye(state_dim, device=state.device)

        # Apply controlled mixing based on entanglement strength
        if qubit1 < state_dim and qubit2 < state_dim:
            mixing_coeff = strength.squeeze() * 0.1  # Small mixing coefficient

            # Cross-connections for entanglement
            mixing_matrix[qubit1, qubit2] = mixing_coeff
            mixing_matrix[qubit2, qubit1] = mixing_coeff

        return torch.matmul(state, mixing_matrix)


class ClassicalAdapter(nn.Module):
    """Standard classical adapter for comparison with quantum approaches"""

    def __init__(self, input_dim: int, output_dim: int, rank: int = 16):
        super().__init__()

        self.rank = rank
        self.down_proj = nn.Linear(input_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, output_dim, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classical adapter"""
        return self.dropout(self.up_proj(self.activation(self.down_proj(x))))


class QuantumAdvantageDetector(nn.Module):
    """
    Detector for identifying scenarios where quantum approaches may provide advantage

    Analyzes input characteristics to predict whether quantum-inspired processing
    would be beneficial over classical approaches.
    """

    def __init__(self, input_dim: int, threshold: float = 0.1):
        super().__init__()

        self.threshold = threshold

        # Feature analyzers
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.correlation_analyzer = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Combination layer
        self.combination_layer = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect potential quantum advantage

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Quantum advantage score [batch_size]
        """
        # Global average pooling
        x_pooled = x.mean(dim=1)  # [batch_size, input_dim]

        # Analyze complexity patterns
        complexity_score = self.complexity_analyzer(x_pooled)

        # Analyze correlation patterns
        correlation_score = self.correlation_analyzer(x_pooled)

        # Combine scores
        combined_features = torch.cat([complexity_score, correlation_score], dim=-1)
        advantage_score = self.combination_layer(combined_features)

        return advantage_score.squeeze(-1)


# Demonstration and testing functions


def demonstrate_quantum_adapters():
    """Demonstrate quantum-inspired adapter functionality"""

    print("⚛️ Quantum-Inspired Adapter Demonstration")
    print("=" * 50)

    # Configuration
    input_dim = 128
    output_dim = 128
    batch_size = 4
    seq_len = 32

    config = QuantumConfig(
        n_qubits=6,
        n_layers=3,
        rotation_gates=["rx", "ry", "rz"],
        entanglement_pattern="linear",
        quantum_advantage_threshold=0.3,
    )

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"Input shape: {x.shape}")
    print(f"Quantum configuration: {config}")

    try:
        # Test Quantum-Inspired Adapter
        print("\n🔬 QUANTUM-INSPIRED ADAPTER:")
        print("-" * 30)

        quantum_adapter = QuantumInspiredAdapter(input_dim, output_dim, config)

        with torch.no_grad():
            quantum_output = quantum_adapter(x)
            print(f"✓ Output shape: {quantum_output.shape}")
            print(
                f"✓ Output statistics: mean={quantum_output.mean():.4f}, std={quantum_output.std():.4f}"
            )

        # Test Hybrid Quantum-Classical Adapter
        print("\n🔄 HYBRID QUANTUM-CLASSICAL ADAPTER:")
        print("-" * 30)

        hybrid_adapter = HybridQuantumClassicalAdapter(input_dim, output_dim, config)

        with torch.no_grad():
            hybrid_output = hybrid_adapter(x)
            print(f"✓ Output shape: {hybrid_output.shape}")
            print(
                f"✓ Output statistics: mean={hybrid_output.mean():.4f}, std={hybrid_output.std():.4f}"
            )

            # Get usage statistics
            usage_stats = hybrid_adapter.get_usage_statistics()
            print(f"✓ Usage statistics: {usage_stats}")

        # Test individual components
        print("\n🧩 COMPONENT TESTING:")
        print("-" * 30)

        # Quantum Feature Map
        feature_map = QuantumFeatureMap(input_dim, config.n_qubits, encoding_type="angle")
        quantum_features = feature_map(x)
        print(f"✓ Quantum feature map output shape: {quantum_features.shape}")

        # Quantum Advantage Detector
        advantage_detector = QuantumAdvantageDetector(input_dim, config.quantum_advantage_threshold)
        advantage_scores = advantage_detector(x)
        print(f"✓ Quantum advantage scores: {advantage_scores}")
        print(f"✓ Average quantum advantage: {advantage_scores.mean():.4f}")

        print("\n" + "=" * 50)
        print("✅ Quantum adapter demonstration completed successfully!")

    except Exception as e:
        print(f"❌ Error in quantum adapter demonstration: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_quantum_adapters()
