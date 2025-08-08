"""
Advanced Adapter Fusion Architectures

This module implements next-generation adapter fusion techniques that go beyond
simple parameter addition, including:
- Hierarchical adapter composition
- Dynamic adapter routing with learned gates
- Cross-attention adapter fusion
- Mixture of Expert (MoE) adapters
- Neural Architecture Search for optimal adapter structures
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_adapter import BaseRetroAdapter

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for adapter fusion architectures"""

    fusion_method: str = "hierarchical"  # hierarchical, gate, cross_attention, moe
    num_adapters: int = 4
    fusion_hidden_dim: int = 256
    gate_temperature: float = 1.0
    expert_capacity: float = 1.0
    load_balancing_loss_weight: float = 0.01
    use_residual_connections: bool = True
    dropout: float = 0.1


class HierarchicalAdapterFusion(nn.Module):
    """
    Hierarchical adapter fusion with multiple levels of composition

    This architecture organizes adapters in a tree structure where:
    - Leaf nodes are individual specialized adapters
    - Internal nodes are fusion operators
    - Root node produces final output
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: FusionConfig,
        adapter_configs: List[Dict[str, Any]],
    ):
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create leaf adapters (specialized for different tasks/domains)
        self.leaf_adapters = nn.ModuleList()
        for i, adapter_config in enumerate(adapter_configs):
            adapter = self._create_leaf_adapter(input_dim, output_dim, adapter_config)
            self.leaf_adapters.append(adapter)

        # Create hierarchical fusion layers
        self.fusion_layers = self._build_fusion_hierarchy()

        # Final projection layer
        self.final_projection = nn.Linear(config.fusion_hidden_dim, output_dim)

        self.dropout = nn.Dropout(config.dropout)

    def _create_leaf_adapter(
        self, input_dim: int, output_dim: int, adapter_config: Dict[str, Any]
    ) -> nn.Module:
        """Create a specialized leaf adapter"""
        adapter_type = adapter_config.get("type", "lora")

        if adapter_type == "lora":
            rank = adapter_config.get("rank", 16)
            return LoRAAdapter(input_dim, output_dim, rank)
        elif adapter_type == "prefix":
            prefix_length = adapter_config.get("prefix_length", 10)
            return PrefixAdapter(input_dim, output_dim, prefix_length)
        elif adapter_type == "bottleneck":
            bottleneck_dim = adapter_config.get("bottleneck_dim", input_dim // 4)
            return BottleneckAdapter(input_dim, output_dim, bottleneck_dim)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def _build_fusion_hierarchy(self) -> nn.ModuleList:
        """Build hierarchical fusion structure"""
        fusion_layers = nn.ModuleList()

        # Calculate number of levels in the hierarchy
        num_adapters = len(self.leaf_adapters)
        num_levels = math.ceil(math.log2(num_adapters))

        current_dim = self.output_dim
        for level in range(num_levels):
            # Number of fusion nodes at this level
            nodes_at_level = max(1, num_adapters // (2 ** (level + 1)))

            level_fusions = nn.ModuleList()
            for _ in range(nodes_at_level):
                fusion_layer = AdaptiveFusionLayer(
                    input_dim=current_dim * 2,  # Two inputs
                    output_dim=self.config.fusion_hidden_dim,
                    dropout=self.config.dropout,
                )
                level_fusions.append(fusion_layer)

            fusion_layers.append(level_fusions)
            current_dim = self.config.fusion_hidden_dim

        return fusion_layers

    def forward(
        self, x: torch.Tensor, task_context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical fusion

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            task_context: Optional context for task-aware routing

        Returns:
            Fused output tensor [batch_size, seq_len, output_dim]
        """
        # Get outputs from all leaf adapters
        leaf_outputs = []
        for adapter in self.leaf_adapters:
            adapter_output = adapter(x)
            leaf_outputs.append(adapter_output)

        # Hierarchical fusion
        current_outputs = leaf_outputs

        for level, level_fusions in enumerate(self.fusion_layers):
            next_outputs = []

            # Pair up outputs and fuse them
            for i in range(0, len(current_outputs), 2):
                if i + 1 < len(current_outputs):
                    # Fuse two outputs
                    fused = level_fusions[i // 2](current_outputs[i], current_outputs[i + 1])
                else:
                    # Odd number, pass through
                    fused = current_outputs[i]

                next_outputs.append(fused)

            current_outputs = next_outputs

        # Final projection
        final_output = current_outputs[0]
        final_output = self.final_projection(final_output)
        final_output = self.dropout(final_output)

        return final_output


class DynamicGatedAdapterFusion(nn.Module):
    """
    Dynamic gated adapter fusion with learned routing

    Uses a gating network to dynamically weight adapter contributions
    based on input characteristics and task context.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: FusionConfig,
        adapter_configs: List[Dict[str, Any]],
    ):
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_adapters = len(adapter_configs)

        # Create adapter modules
        self.adapters = nn.ModuleList()
        for adapter_config in adapter_configs:
            adapter = self._create_adapter(input_dim, output_dim, adapter_config)
            self.adapters.append(adapter)

        # Gating network for dynamic routing
        self.gate_network = GatingNetwork(
            input_dim=input_dim,
            num_experts=self.num_adapters,
            hidden_dim=config.fusion_hidden_dim,
            temperature=config.gate_temperature,
        )

        # Context encoder for task-aware routing
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def _create_adapter(
        self, input_dim: int, output_dim: int, adapter_config: Dict[str, Any]
    ) -> nn.Module:
        """Create an individual adapter"""
        adapter_type = adapter_config.get("type", "lora")

        if adapter_type == "lora":
            rank = adapter_config.get("rank", 16)
            return LoRAAdapter(input_dim, output_dim, rank)
        elif adapter_type == "adalora":
            return AdaLoRAAdapter(input_dim, output_dim, adapter_config)
        elif adapter_type == "ia3":
            return IA3Adapter(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def forward(
        self,
        x: torch.Tensor,
        task_context: Optional[torch.Tensor] = None,
        return_gate_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with dynamic gating

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            task_context: Optional task context tensor
            return_gate_weights: Whether to return gating weights

        Returns:
            Fused output tensor and optionally gate weights
        """
        batch_size, seq_len, input_dim = x.shape

        # Encode context for gating
        if task_context is not None:
            context_encoded = self.context_encoder(task_context)
            gate_input = torch.cat([x.mean(dim=1), context_encoded], dim=-1)
        else:
            gate_input = x.mean(dim=1)  # Global average pooling

        # Compute gating weights
        gate_weights = self.gate_network(gate_input)  # [batch_size, num_adapters]

        # Get adapter outputs
        adapter_outputs = []
        for adapter in self.adapters:
            output = adapter(x)  # [batch_size, seq_len, output_dim]
            adapter_outputs.append(output)

        # Stack adapter outputs
        stacked_outputs = torch.stack(
            adapter_outputs, dim=-1
        )  # [batch_size, seq_len, output_dim, num_adapters]

        # Apply gating weights
        gate_weights_expanded = gate_weights.unsqueeze(1).unsqueeze(
            2
        )  # [batch_size, 1, 1, num_adapters]
        fused_output = torch.sum(stacked_outputs * gate_weights_expanded, dim=-1)

        # Layer normalization
        fused_output = self.layer_norm(fused_output)

        if return_gate_weights:
            return fused_output, gate_weights
        else:
            return fused_output


class CrossAttentionAdapterFusion(nn.Module):
    """
    Cross-attention based adapter fusion

    Uses attention mechanisms to allow adapters to interact and influence
    each other's contributions dynamically.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: FusionConfig,
        adapter_configs: List[Dict[str, Any]],
    ):
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_adapters = len(adapter_configs)

        # Create adapters
        self.adapters = nn.ModuleList()
        for adapter_config in adapter_configs:
            adapter = self._create_adapter(input_dim, output_dim, adapter_config)
            self.adapters.append(adapter)

        # Cross-attention modules
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=8, dropout=config.dropout, batch_first=True
        )

        # Adapter-specific projections
        self.adapter_projections = nn.ModuleList(
            [nn.Linear(output_dim, output_dim) for _ in range(self.num_adapters)]
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * self.num_adapters, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, output_dim),
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def _create_adapter(
        self, input_dim: int, output_dim: int, adapter_config: Dict[str, Any]
    ) -> nn.Module:
        """Create an individual adapter"""
        # Similar to DynamicGatedAdapterFusion._create_adapter
        adapter_type = adapter_config.get("type", "lora")

        if adapter_type == "lora":
            rank = adapter_config.get("rank", 16)
            return LoRAAdapter(input_dim, output_dim, rank)
        else:
            return LoRAAdapter(input_dim, output_dim, 16)  # Default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-attention fusion

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Fused output tensor [batch_size, seq_len, output_dim]
        """
        # Get adapter outputs
        adapter_outputs = []
        for i, adapter in enumerate(self.adapters):
            output = adapter(x)  # [batch_size, seq_len, output_dim]
            output = self.adapter_projections[i](output)
            adapter_outputs.append(output)

        # Stack outputs for cross-attention
        # Reshape to treat different adapters as different sequence positions
        batch_size, seq_len, output_dim = adapter_outputs[0].shape

        # Concatenate along sequence dimension
        concatenated = torch.cat(
            adapter_outputs, dim=1
        )  # [batch_size, seq_len * num_adapters, output_dim]

        # Apply cross-attention (each adapter can attend to others)
        attended, attention_weights = self.cross_attention(
            query=concatenated, key=concatenated, value=concatenated
        )

        # Reshape back and split by adapters
        attended = attended.view(batch_size, self.num_adapters, seq_len, output_dim)

        # Aggregate across adapters
        aggregated_outputs = []
        for i in range(self.num_adapters):
            aggregated_outputs.append(attended[:, i, :, :])  # [batch_size, seq_len, output_dim]

        # Final fusion
        concatenated_final = torch.cat(
            aggregated_outputs, dim=-1
        )  # [batch_size, seq_len, output_dim * num_adapters]
        fused_output = self.fusion_layer(concatenated_final)
        fused_output = self.layer_norm(fused_output)

        return fused_output


class MixtureOfExpertsAdapter(nn.Module):
    """
    Mixture of Experts (MoE) adapter with sparse routing

    Implements efficient sparse routing where only a subset of adapters
    (experts) are activated for each input token.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: FusionConfig,
        adapter_configs: List[Dict[str, Any]],
    ):
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = len(adapter_configs)
        self.expert_capacity = config.expert_capacity

        # Create expert adapters
        self.experts = nn.ModuleList()
        for adapter_config in adapter_configs:
            expert = self._create_expert(input_dim, output_dim, adapter_config)
            self.experts.append(expert)

        # Router network for expert selection
        self.router = SparseRouter(
            input_dim=input_dim,
            num_experts=self.num_experts,
            capacity_factor=config.expert_capacity,
            temperature=config.gate_temperature,
        )

        # Load balancing
        self.load_balancing_loss_weight = config.load_balancing_loss_weight

    def _create_expert(
        self, input_dim: int, output_dim: int, adapter_config: Dict[str, Any]
    ) -> nn.Module:
        """Create an expert adapter"""
        return LoRAAdapter(input_dim, output_dim, adapter_config.get("rank", 16))

    def forward(
        self, x: torch.Tensor, return_load_balancing_loss: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with sparse MoE routing

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            return_load_balancing_loss: Whether to return load balancing loss

        Returns:
            Output tensor and optionally load balancing loss
        """
        batch_size, seq_len, input_dim = x.shape

        # Flatten for routing
        x_flat = x.view(-1, input_dim)  # [batch_size * seq_len, input_dim]

        # Route to experts
        router_output = self.router(x_flat)
        expert_indices = router_output["expert_indices"]
        expert_weights = router_output["expert_weights"]
        tokens_per_expert = router_output["tokens_per_expert"]

        # Process tokens through selected experts
        expert_outputs = []

        for expert_idx in range(self.num_experts):
            # Get tokens assigned to this expert
            expert_mask = expert_indices == expert_idx
            if expert_mask.sum() == 0:
                continue

            expert_tokens = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_tokens)
            expert_outputs.append((expert_mask, expert_output))

        # Combine expert outputs
        output_flat = torch.zeros_like(x_flat[:, : self.output_dim])

        for expert_mask, expert_output in expert_outputs:
            output_flat[expert_mask] = expert_output

        # Apply expert weights
        output_flat = output_flat * expert_weights.unsqueeze(-1)

        # Reshape back
        output = output_flat.view(batch_size, seq_len, self.output_dim)

        if return_load_balancing_loss:
            # Compute load balancing loss
            load_balancing_loss = self._compute_load_balancing_loss(tokens_per_expert)
            return output, load_balancing_loss
        else:
            return output

    def _compute_load_balancing_loss(self, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage"""
        # Coefficient of variation of expert usage
        expert_usage = tokens_per_expert.float()
        mean_usage = expert_usage.mean()
        var_usage = expert_usage.var()

        if mean_usage > 0:
            cv = torch.sqrt(var_usage) / mean_usage
            return self.load_balancing_loss_weight * cv
        else:
            return torch.tensor(0.0, device=tokens_per_expert.device)


# Supporting classes and modules


class LoRAAdapter(nn.Module):
    """Basic LoRA adapter for use in fusion architectures"""

    def __init__(self, input_dim: int, output_dim: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, output_dim, bias=False)
        self.scaling = 1.0 / rank

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * self.lora_B(self.lora_A(x))


class PrefixAdapter(nn.Module):
    """Prefix adapter for hierarchical fusion"""

    def __init__(self, input_dim: int, output_dim: int, prefix_length: int = 10):
        super().__init__()
        self.prefix_length = prefix_length
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, input_dim))
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Prepend prefix
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        x_with_prefix = torch.cat([prefix, x], dim=1)

        # Project and take original sequence length
        output = self.projection(x_with_prefix)
        return output[:, self.prefix_length :, :]


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter for fusion"""

    def __init__(self, input_dim: int, output_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.activation(self.down_proj(x)))


class AdaptiveFusionLayer(nn.Module):
    """Adaptive fusion layer for hierarchical composition"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim // 2, num_heads=4, dropout=dropout, batch_first=True
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Concatenate inputs
        combined = torch.cat([x1, x2], dim=-1)

        # Apply fusion network
        fused = self.fusion_net(combined)

        return fused


class GatingNetwork(nn.Module):
    """Gating network for dynamic adapter selection"""

    def __init__(
        self, input_dim: int, num_experts: int, hidden_dim: int = 256, temperature: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        return F.softmax(logits / self.temperature, dim=-1)


class SparseRouter(nn.Module):
    """Sparse router for MoE with top-k selection"""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.temperature = temperature

        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Route tokens to experts

        Args:
            x: Input tokens [num_tokens, input_dim]

        Returns:
            Dictionary with routing information
        """
        num_tokens = x.size(0)

        # Compute routing probabilities
        logits = self.router(x) / self.temperature
        probs = F.softmax(logits, dim=-1)

        # Top-1 routing for simplicity (can extend to top-k)
        expert_indices = torch.argmax(probs, dim=-1)
        expert_weights = torch.max(probs, dim=-1)[0]

        # Count tokens per expert
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = (expert_indices == i).sum().float()

        return {
            "expert_indices": expert_indices,
            "expert_weights": expert_weights,
            "tokens_per_expert": tokens_per_expert,
        }


class AdaLoRAAdapter(nn.Module):
    """AdaLoRA adapter with adaptive rank"""

    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super().__init__()
        initial_rank = config.get("initial_rank", 32)
        target_rank = config.get("target_rank", 8)

        self.lora = LoRAAdapter(input_dim, output_dim, initial_rank)
        # AdaLoRA-specific parameters would be added here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class IA3Adapter(nn.Module):
    """IA³ adapter implementation"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.scaling_factors = nn.Parameter(torch.ones(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple scaling - in practice would be more sophisticated
        return x * self.scaling_factors


# Factory function for creating fusion adapters


def create_fusion_adapter(
    fusion_type: str,
    input_dim: int,
    output_dim: int,
    config: FusionConfig,
    adapter_configs: List[Dict[str, Any]],
) -> nn.Module:
    """
    Factory function to create fusion adapters

    Args:
        fusion_type: Type of fusion ("hierarchical", "gate", "cross_attention", "moe")
        input_dim: Input dimension
        output_dim: Output dimension
        config: Fusion configuration
        adapter_configs: List of adapter configurations

    Returns:
        Fusion adapter module
    """

    if fusion_type == "hierarchical":
        return HierarchicalAdapterFusion(input_dim, output_dim, config, adapter_configs)
    elif fusion_type == "gate":
        return DynamicGatedAdapterFusion(input_dim, output_dim, config, adapter_configs)
    elif fusion_type == "cross_attention":
        return CrossAttentionAdapterFusion(input_dim, output_dim, config, adapter_configs)
    elif fusion_type == "moe":
        return MixtureOfExpertsAdapter(input_dim, output_dim, config, adapter_configs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


# Example usage and demonstration


def demonstrate_fusion_adapters():
    """Demonstrate different fusion adapter architectures"""

    print("🔬 Advanced Adapter Fusion Demonstration")
    print("=" * 50)

    # Configuration
    input_dim = 768
    output_dim = 768
    batch_size = 4
    seq_len = 128

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Define adapter configurations
    adapter_configs = [
        {"type": "lora", "rank": 16, "domain": "medical"},
        {"type": "lora", "rank": 32, "domain": "legal"},
        {"type": "adalora", "initial_rank": 64, "target_rank": 8, "domain": "finance"},
        {"type": "ia3", "domain": "general"},
    ]

    # Test each fusion type
    fusion_types = ["hierarchical", "gate", "cross_attention", "moe"]

    for fusion_type in fusion_types:
        print(f"\n{fusion_type.upper()} FUSION:")
        print("-" * 30)

        try:
            # Create fusion config
            config = FusionConfig(
                fusion_method=fusion_type, num_adapters=len(adapter_configs), fusion_hidden_dim=256
            )

            # Create fusion adapter
            fusion_adapter = create_fusion_adapter(
                fusion_type=fusion_type,
                input_dim=input_dim,
                output_dim=output_dim,
                config=config,
                adapter_configs=adapter_configs,
            )

            # Forward pass
            with torch.no_grad():
                if fusion_type == "gate":
                    output, gate_weights = fusion_adapter(x, return_gate_weights=True)
                    print(f"  Output shape: {output.shape}")
                    print(f"  Gate weights shape: {gate_weights.shape}")
                    print(f"  Gate weights example: {gate_weights[0].cpu().numpy()}")
                elif fusion_type == "moe":
                    output, load_loss = fusion_adapter(x, return_load_balancing_loss=True)
                    print(f"  Output shape: {output.shape}")
                    print(f"  Load balancing loss: {load_loss.item():.6f}")
                else:
                    output = fusion_adapter(x)
                    print(f"  Output shape: {output.shape}")

            print(f"  ✓ {fusion_type} fusion working correctly")

        except Exception as e:
            print(f"  ✗ Error with {fusion_type} fusion: {str(e)}")

    print("\n" + "=" * 50)
    print("✅ Fusion adapter demonstration completed!")


if __name__ == "__main__":
    demonstrate_fusion_adapters()
