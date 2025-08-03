"""
RetroAdaLoRA: Adaptive LoRA with retrieval-guided rank allocation.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel

from .base_adapter import BaseRetroAdapter


class AdaptiveRetroLoRALayer(nn.Module):
    """
    Adaptive LoRA layer with retrieval-guided rank pruning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_rank: int = 64,
        target_rank: int = 8,
        beta1: float = 0.85,
        beta2: float = 0.85,
        retrieval_importance_weight: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.initial_rank = initial_rank
        self.target_rank = target_rank
        self.current_rank = initial_rank
        self.beta1 = beta1
        self.beta2 = beta2
        self.retrieval_importance_weight = retrieval_importance_weight
        
        # Initialize with maximum rank
        self.lora_A = nn.Linear(in_features, initial_rank, bias=False)
        self.lora_B = nn.Linear(initial_rank, out_features, bias=False)
        
        # Importance scores for rank pruning
        self.register_buffer('importance_scores', torch.ones(initial_rank))
        self.register_buffer('step_count', torch.tensor(0))
        
        # Retrieval enhancement
        self.retrieval_gate = nn.Linear(in_features, initial_rank)
        self.retrieval_scale = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        nn.init.zeros_(self.retrieval_gate.weight)
        nn.init.zeros_(self.retrieval_gate.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        retrieval_context: Optional[torch.Tensor] = None,
        update_importance: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with adaptive rank and retrieval enhancement.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_context: Retrieved context [batch_size, num_docs, context_dim]
            update_importance: Whether to update importance scores
            
        Returns:
            Adaptive LoRA output with retrieval enhancement
        """
        # Forward through LoRA layers
        lora_out = self.lora_A(x)  # [batch_size, seq_len, rank]
        
        # Apply importance masking (zero out unimportant ranks)
        if self.training and update_importance:
            self._update_importance_scores(lora_out)
        
        # Mask using current importance scores
        rank_mask = self._get_rank_mask()
        lora_out = lora_out * rank_mask.unsqueeze(0).unsqueeze(0)
        
        # Retrieval enhancement
        if retrieval_context is not None and retrieval_context.numel() > 0:
            retrieval_enhancement = self._compute_retrieval_enhancement(
                x, retrieval_context, rank_mask
            )
            lora_out = lora_out + retrieval_enhancement
        
        # Final projection
        output = self.lora_B(lora_out)
        
        return output
    
    def _update_importance_scores(self, activations: torch.Tensor):
        """
        Update importance scores based on activation magnitudes.
        
        Args:
            activations: LoRA A output [batch_size, seq_len, rank]
        """
        self.step_count += 1
        
        # Compute current importance (L2 norm across batch and sequence)
        current_importance = torch.norm(activations, dim=(0, 1))
        
        # Exponential moving average update
        if self.step_count == 1:
            self.importance_scores = current_importance
        else:
            self.importance_scores = (
                self.beta1 * self.importance_scores + 
                (1 - self.beta1) * current_importance
            )
    
    def _get_rank_mask(self) -> torch.Tensor:
        """
        Get binary mask for current active ranks.
        
        Returns:
            Binary mask of shape [rank] indicating active ranks
        """
        if self.current_rank >= self.initial_rank:
            return torch.ones_like(self.importance_scores)
        
        # Select top-k most important ranks
        _, top_indices = torch.topk(self.importance_scores, self.current_rank)
        mask = torch.zeros_like(self.importance_scores)
        mask[top_indices] = 1.0
        
        return mask
    
    def _compute_retrieval_enhancement(
        self,
        x: torch.Tensor,
        retrieval_context: torch.Tensor,
        rank_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute retrieval-based enhancement for LoRA activations.
        
        Args:
            x: Input tensor
            retrieval_context: Retrieved context embeddings
            rank_mask: Current rank mask
            
        Returns:
            Retrieval enhancement tensor
        """
        # Pool retrieval context
        pooled_context = retrieval_context.mean(dim=1, keepdim=True)
        
        # Project context to input dimension if needed
        if pooled_context.size(-1) != x.size(-1):
            context_proj = nn.Linear(
                pooled_context.size(-1), 
                x.size(-1),
                device=x.device
            )
            pooled_context = context_proj(pooled_context)
        
        # Expand to sequence length
        pooled_context = pooled_context.expand(-1, x.size(1), -1)
        
        # Compute retrieval gates
        retrieval_gates = torch.sigmoid(self.retrieval_gate(x + pooled_context))
        
        # Apply rank mask to gates
        retrieval_gates = retrieval_gates * rank_mask.unsqueeze(0).unsqueeze(0)
        
        # Scale by retrieval importance
        enhancement = self.retrieval_scale * retrieval_gates
        
        return enhancement
    
    def prune_to_rank(self, target_rank: int):
        """
        Prune adapter to target rank based on importance scores.
        
        Args:
            target_rank: Target rank after pruning
        """
        if target_rank >= self.current_rank:
            return
        
        self.current_rank = min(target_rank, self.initial_rank)
        
        # Get top-k important ranks
        _, top_indices = torch.topk(self.importance_scores, self.current_rank)
        
        # Create pruned layers
        with torch.no_grad():
            # Prune LoRA A
            pruned_A = nn.Linear(self.in_features, self.current_rank, bias=False)
            pruned_A.weight.data = self.lora_A.weight.data[top_indices, :]
            
            # Prune LoRA B  
            pruned_B = nn.Linear(self.current_rank, self.out_features, bias=False)
            pruned_B.weight.data = self.lora_B.weight.data[:, top_indices]
            
            # Prune retrieval gate
            pruned_gate = nn.Linear(self.in_features, self.current_rank)
            pruned_gate.weight.data = self.retrieval_gate.weight.data[top_indices, :]
            pruned_gate.bias.data = self.retrieval_gate.bias.data[top_indices]
            
            # Replace layers
            self.lora_A = pruned_A
            self.lora_B = pruned_B
            self.retrieval_gate = pruned_gate
            
            # Update importance scores
            self.importance_scores = self.importance_scores[top_indices]
    
    def get_rank_utilization(self) -> Dict[str, float]:
        """Get statistics about rank utilization"""
        active_ranks = (self.importance_scores > 0.01).sum().item()
        
        return {
            "current_rank": self.current_rank,
            "active_ranks": active_ranks,
            "utilization": active_ranks / self.current_rank,
            "mean_importance": self.importance_scores.mean().item(),
            "std_importance": self.importance_scores.std().item()
        }


class RetroAdaLoRA(BaseRetroAdapter):
    """
    Adaptive LoRA with retrieval-guided rank allocation.
    
    Dynamically adjusts rank allocation based on:
    1. Activation magnitude importance
    2. Retrieval context relevance  
    3. Task-specific adaptation needs
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        initial_r: int = 64,
        target_r: int = 8,
        beta1: float = 0.85,
        beta2: float = 0.85,
        target_modules: Optional[List[str]] = None,
        retrieval_importance_weight: bool = True,
        rank_update_period: int = 100,
        **kwargs
    ):
        self.initial_r = initial_r
        self.target_r = target_r
        self.beta1 = beta1
        self.beta2 = beta2
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.retrieval_importance_weight = retrieval_importance_weight
        self.rank_update_period = rank_update_period
        self.step_count = 0
        
        super().__init__(base_model=base_model, **kwargs)
    
    def _setup_adapter_layers(self):
        """Setup AdaLoRA layers for target modules"""
        self.ada_lora_layers = nn.ModuleDict()
        
        # Find target modules in the model
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Create adaptive RetroLoRA layer
                    ada_layer = AdaptiveRetroLoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        initial_rank=self.initial_r,
                        target_rank=self.target_r,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        retrieval_importance_weight=self.retrieval_importance_weight
                    )
                    self.ada_lora_layers[name] = ada_layer
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        retrieval_context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive retrieval-augmented LoRA.
        """
        # Get base model embeddings for retrieval if needed
        if retrieval_context is None and self.retriever is not None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            query_embeddings = self.retrieval_projector(inputs_embeds.mean(dim=1))
            retrieval_context, _ = self.retrieve_context(query_embeddings)
        
        # Forward with adaptive layers
        outputs = self._forward_with_ada_lora(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            retrieval_context=retrieval_context,
            **kwargs
        )
        
        # Update step count for rank adaptation
        if self.training:
            self.step_count += 1
            
            # Periodic rank update
            if self.step_count % self.rank_update_period == 0:
                self._update_ranks()
        
        return outputs
    
    def _forward_with_ada_lora(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        retrieval_context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through base model with adaptive LoRA layers.
        """
        original_layers = {}
        
        try:
            # Replace target modules with adaptive LoRA layers
            for name, ada_layer in self.ada_lora_layers.items():
                module_path = name.split('.')
                parent = self.base_model
                for attr in module_path[:-1]:
                    parent = getattr(parent, attr)
                
                original_layers[name] = getattr(parent, module_path[-1])
                
                # Create wrapper that adds adaptive LoRA output
                original_layer = original_layers[name]
                
                def create_ada_wrapper(orig_layer, ada_layer):
                    def ada_forward(x):
                        # Original layer output
                        orig_out = orig_layer(x)
                        # Add adaptive LoRA output
                        ada_out = ada_layer(x, retrieval_context, update_importance=self.training)
                        return orig_out + ada_out
                    return ada_forward
                
                # Set the wrapped forward method
                setattr(parent, module_path[-1], type(original_layer)(
                    original_layer.in_features,
                    original_layer.out_features,
                    bias=original_layer.bias is not None
                ))
                wrapper_layer = getattr(parent, module_path[-1])
                wrapper_layer.weight = original_layer.weight
                if original_layer.bias is not None:
                    wrapper_layer.bias = original_layer.bias
                wrapper_layer.forward = create_ada_wrapper(original_layer, ada_layer)
            
            # Forward through modified model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            
        finally:
            # Restore original layers
            for name, original_layer in original_layers.items():
                module_path = name.split('.')
                parent = self.base_model
                for attr in module_path[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, module_path[-1], original_layer)
        
        return outputs
    
    def _update_ranks(self):
        """
        Update ranks for all adaptive layers based on importance.
        """
        for name, ada_layer in self.ada_lora_layers.items():
            # Compute target rank based on current training progress
            progress = min(1.0, self.step_count / (10 * self.rank_update_period))
            current_target = int(
                self.initial_r - progress * (self.initial_r - self.target_r)
            )
            
            # Prune layer to current target rank
            ada_layer.prune_to_rank(current_target)
    
    def train(
        self,
        dataset,
        importance_metric: str = "retrieval_alignment",
        rank_update_period: int = 100,
        **training_kwargs
    ):
        """
        Train with importance-aware rank adaptation.
        
        Args:
            dataset: Training dataset
            importance_metric: Metric for ranking importance ("activation", "retrieval_alignment")
            rank_update_period: Steps between rank updates
        """
        self.train()
        self.rank_update_period = rank_update_period
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        
        for epoch in range(training_kwargs.get('num_epochs', 3)):
            for batch_idx, batch in enumerate(dataset):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(**batch)
                
                # Compute loss
                loss = outputs.get("loss", torch.tensor(0.0))
                
                # Add rank regularization loss
                rank_loss = self._compute_rank_regularization()
                total_loss = loss + 0.01 * rank_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    rank_stats = self.get_rank_statistics()
                    print(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Loss={loss.item():.4f}, "
                          f"Rank Loss={rank_loss.item():.4f}, "
                          f"Avg Rank={rank_stats['avg_rank']:.1f}")
    
    def _compute_rank_regularization(self) -> torch.Tensor:
        """
        Compute regularization loss to encourage low rank usage.
        """
        total_rank_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for ada_layer in self.ada_lora_layers.values():
            # L1 penalty on rank usage
            rank_usage = ada_layer.current_rank / ada_layer.initial_rank
            total_rank_loss += rank_usage
        
        return total_rank_loss / len(self.ada_lora_layers)
    
    def get_rank_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rank usage statistics"""
        stats = {
            "layer_stats": {},
            "avg_rank": 0.0,
            "total_params": 0,
            "rank_efficiency": 0.0
        }
        
        total_rank = 0
        total_layers = len(self.ada_lora_layers)
        
        for name, ada_layer in self.ada_lora_layers.items():
            layer_stats = ada_layer.get_rank_utilization()
            stats["layer_stats"][name] = layer_stats
            total_rank += layer_stats["current_rank"]
            
            # Count parameters
            layer_params = (
                ada_layer.lora_A.weight.numel() + 
                ada_layer.lora_B.weight.numel() +
                ada_layer.retrieval_gate.weight.numel() +
                ada_layer.retrieval_gate.bias.numel()
            )
            stats["total_params"] += layer_params
        
        if total_layers > 0:
            stats["avg_rank"] = total_rank / total_layers
            stats["rank_efficiency"] = 1.0 - (stats["avg_rank"] / self.initial_r)
        
        return stats
    
    def get_trainable_parameters(self) -> int:
        """Return number of trainable parameters"""
        total_params = 0
        
        for ada_layer in self.ada_lora_layers.values():
            total_params += sum(p.numel() for p in ada_layer.parameters() if p.requires_grad)
        
        # Add retrieval components
        total_params += sum(p.numel() for p in self.retrieval_projector.parameters())
        if hasattr(self, 'context_fusion'):
            total_params += sum(p.numel() for p in self.context_fusion.parameters())
        
        return total_params