"""
RetroLoRA: LoRA adapter with retrieval augmentation.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from .base_adapter import BaseRetroAdapter


class RetroLoRALayer(nn.Module):
    """
    LoRA layer enhanced with retrieval context integration.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        retrieval_rank: int = 8,
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Standard LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Retrieval-enhanced LoRA matrices
        self.retrieval_A = nn.Linear(in_features, retrieval_rank, bias=False)
        self.retrieval_B = nn.Linear(retrieval_rank, out_features, bias=False)
        self.retrieval_gate = nn.Linear(in_features, 1)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.retrieval_A.weight, a=5**0.5)
        nn.init.zeros_(self.retrieval_B.weight)
        nn.init.zeros_(self.retrieval_gate.weight)
        nn.init.zeros_(self.retrieval_gate.bias)

    def forward(
        self, x: torch.Tensor, retrieval_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with retrieval-enhanced LoRA.

        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_context: Retrieved context [batch_size, num_docs, context_dim]

        Returns:
            LoRA output with retrieval enhancement
        """
        # Standard LoRA forward
        lora_out = self.dropout(self.lora_A(x))
        lora_out = self.lora_B(lora_out) * self.scaling

        if retrieval_context is not None and retrieval_context.numel() > 0:
            # Compute retrieval gate
            gate = torch.sigmoid(self.retrieval_gate(x))

            # Pool retrieval context (mean over documents)
            pooled_context = retrieval_context.mean(dim=1, keepdim=True)
            if pooled_context.size(-1) != x.size(-1):
                # Project context to input dimension
                context_proj = nn.Linear(pooled_context.size(-1), x.size(-1), device=x.device)
                pooled_context = context_proj(pooled_context)

            # Expand to sequence length
            pooled_context = pooled_context.expand(-1, x.size(1), -1)

            # Retrieval-enhanced LoRA
            retrieval_input = x + 0.1 * pooled_context
            retrieval_out = self.dropout(self.retrieval_A(retrieval_input))
            retrieval_out = self.retrieval_B(retrieval_out)

            # Gated combination
            lora_out = lora_out + gate * retrieval_out

        return lora_out


class RetroLoRA(BaseRetroAdapter):
    """
    Low-Rank Adaptation with retrieval augmentation.

    Combines standard LoRA with retrieval-enhanced adaptation for
    efficient domain-specific fine-tuning.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        retrieval_rank: int = 8,
        **kwargs,
    ):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.retrieval_rank = retrieval_rank

        super().__init__(base_model=base_model, **kwargs)

    def _setup_adapter_layers(self):
        """Setup RetroLoRA layers for target modules"""
        self.retro_lora_layers = nn.ModuleDict()

        # Find target modules in the model
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with RetroLoRA layer
                    retro_layer = RetroLoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.r,
                        alpha=self.alpha,
                        dropout=self.dropout,
                        retrieval_rank=self.retrieval_rank,
                    )
                    self.retro_lora_layers[name] = retro_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        retrieval_context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with retrieval-enhanced LoRA.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for training [batch_size, seq_len]
            retrieval_context: Retrieved context embeddings [batch_size, num_docs, dim]

        Returns:
            Dictionary with model outputs
        """
        # Get base model embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # If no retrieval context provided, try to retrieve
        if retrieval_context is None and self.retriever is not None:
            # Use input embeddings as query for retrieval
            query_embeddings = self.retrieval_projector(inputs_embeds.mean(dim=1))
            retrieval_context, _ = self.retrieve_context(query_embeddings)

        # Apply retrieval-enhanced LoRA to base model
        outputs = self._forward_with_retro_lora(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            retrieval_context=retrieval_context,
            **kwargs,
        )

        return outputs

    def _forward_with_retro_lora(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        retrieval_context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through base model with RetroLoRA layers applied.
        """
        # Store original linear layers
        original_layers = {}

        try:
            # Replace target modules with RetroLoRA layers
            for name, retro_layer in self.retro_lora_layers.items():
                module_path = name.split(".")
                parent = self.base_model
                for attr in module_path[:-1]:
                    parent = getattr(parent, attr)

                original_layers[name] = getattr(parent, module_path[-1])

                # Create wrapper that adds RetroLoRA output
                original_layer = original_layers[name]

                def create_retro_wrapper(orig_layer, retro_layer):
                    def retro_forward(x):
                        # Original layer output
                        orig_out = orig_layer(x)
                        # Add RetroLoRA output
                        retro_out = retro_layer(x, retrieval_context)
                        return orig_out + retro_out

                    return retro_forward

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
                wrapper_layer.forward = create_retro_wrapper(original_layer, retro_layer)

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

    def train_with_retrieval(
        self,
        dataset,
        num_epochs: int = 3,
        retrieval_k: int = 5,
        retrieval_weight: float = 0.3,
        negative_sampling: bool = True,
        hard_negatives_ratio: float = 0.5,
        **training_kwargs,
    ):
        """
        Train RetroLoRA with retrieval supervision.

        Args:
            dataset: Training dataset with text and optional retrieval targets
            num_epochs: Number of training epochs
            retrieval_k: Number of documents to retrieve per query
            retrieval_weight: Weight for retrieval loss component
            negative_sampling: Whether to use negative sampling for retrieval
            hard_negatives_ratio: Ratio of hard negatives in sampling
        """
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_gen_loss = 0.0
            total_ret_loss = 0.0

            for batch_idx, batch in enumerate(dataset):
                optimizer.zero_grad()

                # Prepare batch with retrieval context if needed
                if self.retriever is not None and "retrieval_context" not in batch:
                    # Retrieve context for this batch
                    input_embeds = self.base_model.get_input_embeddings()(batch["input_ids"])
                    query_embeddings = self.retrieval_projector(input_embeds.mean(dim=1))
                    retrieval_context, _ = self.retrieve_context(query_embeddings)
                    batch["retrieval_context"] = retrieval_context

                # Training step
                loss_dict = self.train_step(batch, retrieval_weight)

                # Backward pass
                loss_dict["loss"].backward()
                optimizer.step()

                # Accumulate losses
                total_loss += loss_dict["loss"].item()
                total_gen_loss += loss_dict["generation_loss"].item()
                total_ret_loss += loss_dict["retrieval_loss"].item()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"Loss={loss_dict['loss'].item():.4f}, "
                        f"Gen={loss_dict['generation_loss'].item():.4f}, "
                        f"Ret={loss_dict['retrieval_loss'].item():.4f}"
                    )

            avg_loss = total_loss / len(dataset)
            avg_gen_loss = total_gen_loss / len(dataset)
            avg_ret_loss = total_ret_loss / len(dataset)

            print(
                f"Epoch {epoch} completed: "
                f"Avg Loss={avg_loss:.4f}, "
                f"Gen Loss={avg_gen_loss:.4f}, "
                f"Ret Loss={avg_ret_loss:.4f}"
            )

    def merge_adapter(self):
        """
        Merge RetroLoRA weights into base model for deployment.

        This creates a single model with RetroLoRA weights merged,
        reducing inference overhead.
        """
        with torch.no_grad():
            for name, retro_layer in self.retro_lora_layers.items():
                # Get target module
                module_path = name.split(".")
                parent = self.base_model
                for attr in module_path[:-1]:
                    parent = getattr(parent, attr)
                target_module = getattr(parent, module_path[-1])

                if isinstance(target_module, nn.Linear):
                    # Compute merged LoRA weights
                    lora_weight = retro_layer.lora_B.weight @ retro_layer.lora_A.weight
                    merged_weight = target_module.weight + retro_layer.scaling * lora_weight

                    # Update target module weights
                    target_module.weight.data = merged_weight

        print("RetroLoRA adapter merged into base model")

    def get_trainable_parameters(self) -> int:
        """Return number of trainable parameters"""
        total_params = 0
        for retro_layer in self.retro_lora_layers.values():
            total_params += sum(p.numel() for p in retro_layer.parameters() if p.requires_grad)

        # Add retrieval components
        total_params += sum(p.numel() for p in self.retrieval_projector.parameters())
        if hasattr(self, "context_fusion"):
            total_params += sum(p.numel() for p in self.context_fusion.parameters())

        return total_params
