"""
RetroIA3: (IA)³ adapters with retrieval scaling.

Implements Infused Adapter by Inhibiting and Amplifying Inner Activations (IA³)
enhanced with retrieval-based context scaling.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .base_adapter import BaseRetroAdapter


class RetroIA3Layer(nn.Module):
    """
    IA³ layer with retrieval-based scaling.

    IA³ applies learned scaling factors to key and value projections
    and feed-forward networks, enhanced with retrieval context.
    """

    def __init__(
        self,
        hidden_size: int,
        layer_type: str = "attention",  # "attention" or "feedforward"
        retrieval_scale_factor: float = 2.0,
        init_ia3_weights: str = "xavier",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_type = layer_type
        self.retrieval_scale_factor = retrieval_scale_factor

        # IA³ scaling parameters
        if layer_type == "attention":
            # Scale for key and value projections
            self.scale_k = nn.Parameter(torch.ones(hidden_size))
            self.scale_v = nn.Parameter(torch.ones(hidden_size))
        elif layer_type == "feedforward":
            # Scale for feed-forward intermediate layer
            self.scale_ff = nn.Parameter(torch.ones(hidden_size))
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")

        # Retrieval-based enhancement
        self.retrieval_projector = nn.Linear(hidden_size, hidden_size, bias=False)
        self.retrieval_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())

        # Initialize weights
        self._init_weights(init_ia3_weights)

    def _init_weights(self, init_method: str):
        """Initialize IA³ weights"""
        if init_method == "xavier":
            if hasattr(self, "scale_k"):
                nn.init.xavier_uniform_(self.scale_k.unsqueeze(0)).squeeze_(0)
                nn.init.xavier_uniform_(self.scale_v.unsqueeze(0)).squeeze_(0)
            if hasattr(self, "scale_ff"):
                nn.init.xavier_uniform_(self.scale_ff.unsqueeze(0)).squeeze_(0)
        elif init_method == "ones":
            # Keep default initialization (ones)
            pass
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # Initialize retrieval components
        nn.init.xavier_uniform_(self.retrieval_projector.weight)
        for layer in self.retrieval_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        retrieval_context: Optional[torch.Tensor] = None,
        layer_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply IA³ scaling with retrieval enhancement.

        Args:
            x: Input tensor (K, V for attention or intermediate for FF)
            retrieval_context: Retrieved context embeddings
            layer_input: Original layer input for gate computation

        Returns:
            Scaled output tensor
        """
        # Base IA³ scaling
        if self.layer_type == "attention":
            # Assume x is concatenated [key, value]
            # Split into key and value components
            if x.size(-1) == 2 * self.hidden_size:
                key, value = torch.chunk(x, 2, dim=-1)
                scaled_key = key * self.scale_k
                scaled_value = value * self.scale_v
                scaled_x = torch.cat([scaled_key, scaled_value], dim=-1)
            else:
                # Apply to single tensor (either K or V)
                if hasattr(self, "scale_k"):
                    scaled_x = x * self.scale_k
                else:
                    scaled_x = x * self.scale_v
        elif self.layer_type == "feedforward":
            scaled_x = x * self.scale_ff

        # Retrieval enhancement
        if retrieval_context is not None and retrieval_context.numel() > 0:
            scaled_x = self._apply_retrieval_enhancement(scaled_x, retrieval_context, layer_input)

        return scaled_x

    def _apply_retrieval_enhancement(
        self,
        scaled_x: torch.Tensor,
        retrieval_context: torch.Tensor,
        layer_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply retrieval-based enhancement to IA³ output.

        Args:
            scaled_x: IA³ scaled tensor
            retrieval_context: Retrieved context embeddings
            layer_input: Original layer input for gating

        Returns:
            Enhanced tensor with retrieval information
        """
        batch_size, seq_len = scaled_x.shape[:2]

        # Pool retrieval context
        pooled_context = retrieval_context.mean(dim=1)  # [batch_size, context_dim]

        # Project context to match hidden dimension
        if pooled_context.size(-1) != self.hidden_size:
            context_proj = nn.Linear(
                pooled_context.size(-1), self.hidden_size, device=scaled_x.device
            )
            pooled_context = context_proj(pooled_context)

        # Project retrieval context
        projected_context = self.retrieval_projector(pooled_context)

        # Expand to sequence length
        expanded_context = projected_context.unsqueeze(1).expand(-1, seq_len, -1)

        # Compute retrieval gate
        if layer_input is not None:
            gate_input = torch.cat([scaled_x, expanded_context], dim=-1)
            gate = self.retrieval_gate(gate_input)

            # Apply gated enhancement
            enhancement = gate * expanded_context * self.retrieval_scale_factor
            enhanced_x = scaled_x + enhancement
        else:
            # Simple additive enhancement
            enhancement = expanded_context * self.retrieval_scale_factor
            enhanced_x = scaled_x + 0.1 * enhancement

        return enhanced_x


class RetroIA3(BaseRetroAdapter):
    """
    Infused Adapter by Inhibiting and Amplifying Inner Activations (IA³)
    with retrieval enhancement.

    IA³ is highly parameter-efficient, requiring only ~0.01% additional parameters
    while maintaining competitive performance through strategic activation scaling.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        target_modules: Optional[List[str]] = None,
        retrieval_scale_factor: float = 2.0,
        init_ia3_weights: str = "xavier",
        **kwargs,
    ):
        self.target_modules = target_modules or ["k_proj", "v_proj", "down_proj"]
        self.retrieval_scale_factor = retrieval_scale_factor
        self.init_ia3_weights = init_ia3_weights

        super().__init__(base_model=base_model, **kwargs)

    def _setup_adapter_layers(self):
        """Setup IA³ layers for target modules"""
        self.ia3_layers = nn.ModuleDict()

        # Find target modules in the model
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Determine layer type
                    if any(kv in name for kv in ["k_proj", "v_proj", "q_proj"]):
                        layer_type = "attention"
                    elif any(ff in name for ff in ["down_proj", "up_proj", "gate_proj"]):
                        layer_type = "feedforward"
                    else:
                        layer_type = "attention"  # default

                    # Create IA³ layer
                    ia3_layer = RetroIA3Layer(
                        hidden_size=module.out_features,
                        layer_type=layer_type,
                        retrieval_scale_factor=self.retrieval_scale_factor,
                        init_ia3_weights=self.init_ia3_weights,
                    )
                    self.ia3_layers[name] = ia3_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        retrieval_context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with IA³ retrieval enhancement.
        """
        # Get retrieval context if needed
        if retrieval_context is None and self.retriever is not None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            query_embeddings = self.retrieval_projector(inputs_embeds.mean(dim=1))
            retrieval_context, _ = self.retrieve_context(query_embeddings)

        # Forward with IA³ layers
        outputs = self._forward_with_ia3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            retrieval_context=retrieval_context,
            **kwargs,
        )

        return outputs

    def _forward_with_ia3(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        retrieval_context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through base model with IA³ layers applied.
        """
        original_layers = {}

        try:
            # Replace target modules with IA³-enhanced versions
            for name, ia3_layer in self.ia3_layers.items():
                module_path = name.split(".")
                parent = self.base_model
                for attr in module_path[:-1]:
                    parent = getattr(parent, attr)

                original_layers[name] = getattr(parent, module_path[-1])

                # Create wrapper that applies IA³ scaling
                original_layer = original_layers[name]

                def create_ia3_wrapper(orig_layer, ia3_layer):
                    def ia3_forward(x):
                        # Original layer forward
                        orig_out = orig_layer(x)

                        # Apply IA³ scaling with retrieval enhancement
                        scaled_out = ia3_layer(
                            orig_out, retrieval_context=retrieval_context, layer_input=x
                        )

                        return scaled_out

                    return ia3_forward

                # Set the wrapped forward method
                setattr(
                    parent,
                    module_path[-1],
                    type(original_layer)(
                        original_layer.in_features,
                        original_layer.out_features,
                        bias=original_layer.bias is not None,
                    ),
                )
                wrapper_layer = getattr(parent, module_path[-1])
                wrapper_layer.weight = original_layer.weight
                if original_layer.bias is not None:
                    wrapper_layer.bias = original_layer.bias
                wrapper_layer.forward = create_ia3_wrapper(original_layer, ia3_layer)

            # Forward through modified model
            outputs = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
            )

        finally:
            # Restore original layers
            for name, original_layer in original_layers.items():
                module_path = name.split(".")
                parent = self.base_model
                for attr in module_path[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, module_path[-1], original_layer)

        return outputs

    def few_shot_train(
        self,
        support_set: List[Dict[str, Any]],
        retrieval_augmentation: bool = True,
        meta_learning_rate: float = 0.001,
        num_inner_steps: int = 5,
        **kwargs,
    ):
        """
        Few-shot learning with IA³ and retrieval augmentation.

        Args:
            support_set: List of few-shot examples
            retrieval_augmentation: Whether to use retrieval
            meta_learning_rate: Learning rate for inner loop
            num_inner_steps: Number of gradient steps per task
        """
        self.train()

        # Meta-optimizer for IA³ parameters
        meta_optimizer = torch.optim.Adam(self.parameters(), lr=meta_learning_rate)

        for epoch in range(kwargs.get("num_epochs", 10)):
            total_loss = 0.0

            for task_data in support_set:
                # Save current parameters
                original_params = {name: param.clone() for name, param in self.named_parameters()}

                # Inner loop: adapt to current task
                task_optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

                for step in range(num_inner_steps):
                    task_optimizer.zero_grad()

                    # Prepare batch
                    if retrieval_augmentation and self.retriever is not None:
                        # Add retrieval context to task data
                        inputs_embeds = self.base_model.get_input_embeddings()(
                            task_data["input_ids"]
                        )
                        query_embeddings = self.retrieval_projector(inputs_embeds.mean(dim=1))
                        retrieval_context, _ = self.retrieve_context(query_embeddings)
                        task_data["retrieval_context"] = retrieval_context

                    # Forward pass
                    outputs = self.forward(**task_data)
                    loss = outputs.get("loss", torch.tensor(0.0))

                    # Backward pass
                    loss.backward()
                    task_optimizer.step()

                # Compute meta-loss on query set (use same data for simplicity)
                meta_optimizer.zero_grad()

                outputs = self.forward(**task_data)
                meta_loss = outputs.get("loss", torch.tensor(0.0))

                # Meta-gradient step
                meta_loss.backward()
                meta_optimizer.step()

                total_loss += meta_loss.item()

                # Restore parameters for next task
                for name, param in self.named_parameters():
                    param.data = original_params[name]

            avg_loss = total_loss / len(support_set)
            print(f"Few-shot Epoch {epoch}: Meta Loss = {avg_loss:.4f}")

    def merge_adapter(self):
        """
        Merge IA³ scaling factors into base model weights.

        Since IA³ only scales activations, merging involves incorporating
        the scaling factors directly into the weight matrices.
        """
        with torch.no_grad():
            for name, ia3_layer in self.ia3_layers.items():
                # Get target module
                module_path = name.split(".")
                parent = self.base_model
                for attr in module_path[:-1]:
                    parent = getattr(parent, attr)
                target_module = getattr(parent, module_path[-1])

                if isinstance(target_module, nn.Linear):
                    # Apply IA³ scaling to weights
                    if ia3_layer.layer_type == "attention":
                        if hasattr(ia3_layer, "scale_k") and "k_proj" in name:
                            target_module.weight.data *= ia3_layer.scale_k.unsqueeze(0)
                        elif hasattr(ia3_layer, "scale_v") and "v_proj" in name:
                            target_module.weight.data *= ia3_layer.scale_v.unsqueeze(0)
                    elif ia3_layer.layer_type == "feedforward":
                        target_module.weight.data *= ia3_layer.scale_ff.unsqueeze(0)

        print("IA³ adapter merged into base model")

    def get_trainable_parameters(self) -> int:
        """Return number of trainable parameters"""
        total_params = 0

        for ia3_layer in self.ia3_layers.values():
            total_params += sum(p.numel() for p in ia3_layer.parameters() if p.requires_grad)

        # Add retrieval components
        total_params += sum(p.numel() for p in self.retrieval_projector.parameters())
        if hasattr(self, "context_fusion"):
            total_params += sum(p.numel() for p in self.context_fusion.parameters())

        return total_params

    def get_parameter_efficiency(self) -> Dict[str, Any]:
        """Get parameter efficiency statistics"""
        adapter_params = self.get_trainable_parameters()

        # Count base model parameters
        base_params = sum(p.numel() for p in self.base_model.parameters())

        efficiency = adapter_params / base_params

        return {
            "adapter_parameters": adapter_params,
            "base_model_parameters": base_params,
            "efficiency_ratio": efficiency,
            "efficiency_percentage": efficiency * 100,
            "parameter_reduction": (1 - efficiency) * 100,
        }
