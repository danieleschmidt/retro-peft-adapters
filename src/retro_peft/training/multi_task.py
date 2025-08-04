"""
Multi-task training for retrieval-augmented adapters.

Enables joint training on multiple objectives including generation,
retrieval alignment, and domain classification.
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from ..adapters.base_adapter import BaseRetroAdapter


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task training with different task types.
    
    Supports generation, retrieval, and classification tasks
    with dynamic task sampling strategies.
    """
    
    def __init__(
        self,
        task_datasets: Dict[str, List[Dict[str, Any]]],
        tokenizer,
        max_length: int = 512,
        task_sampling: str = "proportional"  # "proportional", "uniform", "curriculum"
    ):
        """
        Initialize multi-task dataset.
        
        Args:
            task_datasets: Dictionary mapping task names to datasets
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            task_sampling: Strategy for sampling tasks
        """
        self.task_datasets = task_datasets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_sampling = task_sampling
        
        # Build unified dataset with task labels
        self.unified_data = []
        self.task_indices = {}
        
        for task_name, task_data in task_datasets.items():
            start_idx = len(self.unified_data)
            
            for item in task_data:
                unified_item = item.copy()
                unified_item["task_name"] = task_name
                self.unified_data.append(unified_item)
            
            end_idx = len(self.unified_data)
            self.task_indices[task_name] = (start_idx, end_idx)
        
        # Task sampling weights
        self.task_weights = self._compute_task_weights()
        self.current_epoch = 0
    
    def _compute_task_weights(self) -> Dict[str, float]:
        """Compute sampling weights for tasks"""
        if self.task_sampling == "uniform":
            # Equal weight for all tasks
            num_tasks = len(self.task_datasets)
            return {task: 1.0 / num_tasks for task in self.task_datasets.keys()}
        
        elif self.task_sampling == "proportional":
            # Weight proportional to dataset size
            total_size = sum(len(data) for data in self.task_datasets.values())
            return {
                task: len(data) / total_size 
                for task, data in self.task_datasets.items()
            }
        
        elif self.task_sampling == "curriculum":
            # Will be updated dynamically during training
            return {task: 1.0 for task in self.task_datasets.keys()}
        
        else:
            raise ValueError(f"Unknown task sampling strategy: {self.task_sampling}")
    
    def update_task_weights(self, new_weights: Dict[str, float]):
        """Update task sampling weights (for curriculum learning)"""
        self.task_weights = new_weights
    
    def set_epoch(self, epoch: int):
        """Set current epoch (for curriculum learning)"""
        self.current_epoch = epoch
    
    def __len__(self):
        return len(self.unified_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.unified_data[idx]
        task_name = item["task_name"]
        
        # Process based on task type
        if task_name.endswith("_generation"):
            return self._process_generation_item(item)
        elif task_name.endswith("_retrieval"):
            return self._process_retrieval_item(item)
        elif task_name.endswith("_classification"):
            return self._process_classification_item(item)
        else:
            # Default: treat as generation task
            return self._process_generation_item(item)
    
    def _process_generation_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process generation task item"""
        input_text = item.get("input_text", "")
        target_text = item.get("target_text", "")
        
        # Combine input and target for language modeling
        full_text = input_text + " " + target_text
        
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (shift by one for language modeling)
        input_ids = tokens["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask input portion in labels
        input_length = len(self.tokenizer.encode(input_text, add_special_tokens=False))
        labels[:input_length] = -100  # Ignore loss for input portion
        
        return {
            "task_name": item["task_name"],
            "task_type": "generation",
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "retrieval_context": item.get("retrieval_context"),
            "metadata": item.get("metadata", {})
        }
    
    def _process_retrieval_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process retrieval alignment task item"""
        query_text = item.get("query_text", "")
        positive_doc = item.get("positive_doc", "")
        
        query_tokens = self.tokenizer(
            query_text,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        doc_tokens = self.tokenizer(
            positive_doc,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "task_name": item["task_name"],
            "task_type": "retrieval",
            "query_input_ids": query_tokens["input_ids"].squeeze(0),
            "query_attention_mask": query_tokens["attention_mask"].squeeze(0),
            "doc_input_ids": doc_tokens["input_ids"].squeeze(0),
            "doc_attention_mask": doc_tokens["attention_mask"].squeeze(0),
            "retrieval_label": torch.tensor(1.0),  # Positive pair
            "metadata": item.get("metadata", {})
        }
    
    def _process_classification_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process classification task item"""
        text = item.get("text", "")
        label = item.get("label", 0)
        
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "task_name": item["task_name"],
            "task_type": "classification",
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "classification_label": torch.tensor(label, dtype=torch.long),
            "metadata": item.get("metadata", {})
        }
    
    def sample_by_task(self, batch_size: int) -> List[int]:
        """Sample indices based on task weights"""
        indices = []
        
        for _ in range(batch_size):
            # Sample task based on weights
            task_names = list(self.task_weights.keys())
            task_probs = list(self.task_weights.values())
            
            selected_task = np.random.choice(task_names, p=task_probs)
            
            # Sample from selected task
            start_idx, end_idx = self.task_indices[selected_task]
            task_idx = random.randint(start_idx, end_idx - 1)
            indices.append(task_idx)
        
        return indices


class MultiTaskRetroTrainer:
    """
    Multi-task trainer for retrieval-augmented adapters.
    
    Supports joint training on generation, retrieval, and classification
    tasks with flexible loss weighting and curriculum learning.
    """
    
    def __init__(
        self,
        model: BaseRetroAdapter,
        tasks: Dict[str, Dict[str, Union[float, str]]],
        gradient_accumulation_steps: int = 1,
        task_sampling: str = "proportional",
        temperature: float = 0.07
    ):
        """
        Initialize multi-task trainer.
        
        Args:
            model: Retrieval-augmented adapter to train
            tasks: Dictionary defining tasks with weights and loss types
            gradient_accumulation_steps: Steps to accumulate gradients
            task_sampling: Task sampling strategy
            temperature: Temperature for contrastive losses
        """
        self.model = model
        self.tasks = tasks
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.task_sampling = task_sampling
        self.temperature = temperature
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Task-specific components
        self._setup_task_heads()
        
        # Loss functions
        self.loss_functions = self._setup_loss_functions()
    
    def _setup_task_heads(self):
        """Setup task-specific prediction heads"""
        self.task_heads = nn.ModuleDict()
        
        for task_name, task_config in self.tasks.items():
            if task_name.endswith("_classification"):
                # Classification head
                num_classes = task_config.get("num_classes", 2)
                self.task_heads[task_name] = nn.Linear(
                    self.model.base_model.config.hidden_size,
                    num_classes
                )
            elif task_name.endswith("_retrieval"):
                # Retrieval alignment head (already in base model)
                pass
            # Generation uses the base model's language modeling head
        
        # Move to device
        self.task_heads = self.task_heads.to(self.device)
    
    def _setup_loss_functions(self) -> Dict[str, Callable]:
        """Setup loss functions for different tasks"""
        loss_functions = {}
        
        for task_name, task_config in self.tasks.items():
            loss_type = task_config.get("loss", "cross_entropy")
            
            if loss_type == "cross_entropy":
                loss_functions[task_name] = nn.CrossEntropyLoss(ignore_index=-100)
            elif loss_type == "mse":
                loss_functions[task_name] = nn.MSELoss()
            elif loss_type == "contrastive":
                loss_functions[task_name] = self._contrastive_loss
            elif loss_type == "bce":
                loss_functions[task_name] = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss_functions
    
    def train(
        self,
        datasets: Dict[str, List[Dict[str, Any]]],
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        eval_steps: int = 500,
        save_steps: int = 1000,
        output_dir: str = "./multitask_checkpoints",
        curriculum_schedule: Optional[Dict[int, Dict[str, float]]] = None
    ):
        """
        Train with multi-task learning.
        
        Args:
            datasets: Dictionary mapping task names to datasets
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate
            eval_steps: Steps between evaluations
            save_steps: Steps between saves
            output_dir: Output directory for checkpoints
            curriculum_schedule: Optional curriculum learning schedule
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create multi-task dataset
        dataset = MultiTaskDataset(
            task_datasets=datasets,
            tokenizer=getattr(self.model.base_model, 'tokenizer', None),
            task_sampling=self.task_sampling
        )
        
        if dataset.tokenizer is None:
            raise ValueError("Model must have tokenizer attribute for multi-task training")
        
        # Setup data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.task_heads.parameters()),
            lr=learning_rate
        )
        
        total_steps = len(dataloader) * num_epochs
        scheduler = self._get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.tasks.keys()}
        
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Update curriculum if specified
            if curriculum_schedule and epoch in curriculum_schedule:
                new_weights = curriculum_schedule[epoch]
                dataset.update_task_weights(new_weights)
                print(f"Updated task weights: {new_weights}")
            
            dataset.set_epoch(epoch)
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute multi-task loss
                loss_dict = self._compute_multi_task_loss(batch)
                total_batch_loss = loss_dict["total_loss"]
                
                # Scale loss by gradient accumulation steps
                scaled_loss = total_batch_loss / self.gradient_accumulation_steps
                scaled_loss.backward()
                
                # Update metrics
                total_loss += total_batch_loss.item()
                epoch_loss += total_batch_loss.item()
                
                for task_name, task_loss in loss_dict.get("task_losses", {}).items():
                    task_losses[task_name] += task_loss
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.task_heads.parameters()), 
                        1.0
                    )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # Logging
                if global_step % 100 == 0:
                    avg_loss = total_loss / max(global_step, 1)
                    current_lr = scheduler.get_last_lr()[0]
                    
                    task_loss_str = ", ".join([
                        f"{task}={loss/max(global_step, 1):.4f}"
                        for task, loss in task_losses.items()
                    ])
                    
                    print(f"Epoch {epoch}, Step {global_step}: "
                          f"Total Loss={total_batch_loss.item():.4f}, "
                          f"Avg Loss={avg_loss:.4f}, "
                          f"Task Losses=[{task_loss_str}], "
                          f"LR={current_lr:.2e}")
                
                # Evaluation
                if global_step % eval_steps == 0:
                    eval_metrics = self._evaluate(datasets)
                    print(f"Evaluation at step {global_step}: {eval_metrics}")
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_path = os.path.join(
                        output_dir, f"checkpoint-{global_step}"
                    )
                    self._save_checkpoint(checkpoint_path, global_step)
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Final save
        final_path = os.path.join(output_dir, "final_model")
        self._save_checkpoint(final_path, global_step)
        print(f"Training completed. Final model saved to: {final_path}")
    
    def _compute_multi_task_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss for batch"""
        task_losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Group batch by task type
        task_batches = self._group_batch_by_task(batch)
        
        for task_name, task_batch in task_batches.items():
            if not task_batch:  # Skip empty batches
                continue
            
            task_config = self.tasks[task_name]
            task_weight = task_config.get("weight", 1.0)
            
            # Compute task-specific loss
            if task_name.endswith("_generation"):
                task_loss = self._compute_generation_loss(task_batch)
            elif task_name.endswith("_retrieval"):
                task_loss = self._compute_retrieval_loss(task_batch)
            elif task_name.endswith("_classification"):
                task_loss = self._compute_classification_loss(task_batch, task_name)
            else:
                # Default to generation
                task_loss = self._compute_generation_loss(task_batch)
            
            # Weight and accumulate
            weighted_loss = task_weight * task_loss
            total_loss += weighted_loss
            task_losses[task_name] = task_loss.item()
        
        return {
            "total_loss": total_loss,
            "task_losses": task_losses
        }
    
    def _group_batch_by_task(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Group batch items by task name"""
        task_batches = {}
        batch_size = len(batch["task_name"])
        
        for i in range(batch_size):
            task_name = batch["task_name"][i]
            
            if task_name not in task_batches:
                task_batches[task_name] = {}
            
            # Add this item to the task batch
            for key, value in batch.items():
                if key == "task_name":
                    continue
                
                if key not in task_batches[task_name]:
                    task_batches[task_name][key] = []
                
                if isinstance(value, torch.Tensor):
                    task_batches[task_name][key].append(value[i])
                else:
                    task_batches[task_name][key].append(value[i])
        
        # Stack tensors
        for task_name, task_batch in task_batches.items():
            for key, value_list in task_batch.items():
                if isinstance(value_list[0], torch.Tensor):
                    task_batches[task_name][key] = torch.stack(value_list)
        
        return task_batches
    
    def _compute_generation_loss(self, task_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute generation loss"""
        # Forward through model
        outputs = self.model(
            input_ids=task_batch["input_ids"],
            attention_mask=task_batch["attention_mask"],
            labels=task_batch["labels"],
            retrieval_context=task_batch.get("retrieval_context")
        )
        
        return outputs.get("loss", torch.tensor(0.0, device=self.device))
    
    def _compute_retrieval_loss(self, task_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute retrieval alignment loss"""
        # Encode query and document
        query_embeddings = self._encode_texts(
            task_batch["query_input_ids"],
            task_batch["query_attention_mask"]
        )
        
        doc_embeddings = self._encode_texts(
            task_batch["doc_input_ids"],
            task_batch["doc_attention_mask"]
        )
        
        # Compute contrastive loss
        return self._contrastive_loss(query_embeddings, doc_embeddings)
    
    def _compute_classification_loss(
        self, 
        task_batch: Dict[str, torch.Tensor],
        task_name: str
    ) -> torch.Tensor:
        """Compute classification loss"""
        # Get embeddings
        embeddings = self._encode_texts(
            task_batch["input_ids"],
            task_batch["attention_mask"]
        )
        
        # Forward through task-specific head
        logits = self.task_heads[task_name](embeddings)
        
        # Compute loss
        loss_fn = self.loss_functions[task_name]
        return loss_fn(logits, task_batch["classification_label"])
    
    def _encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode texts using model"""
        # Get embeddings from base model
        embeddings = self.model.base_model.get_input_embeddings()(input_ids)
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_embeddings = sum_embeddings / sum_mask
        
        # Project to retrieval dimension
        return self.model.retrieval_projector(pooled_embeddings)
    
    def _contrastive_loss(
        self, 
        query_embeddings: torch.Tensor, 
        doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between queries and documents"""
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.mm(query_embeddings, doc_embeddings.t()) / self.temperature
        
        # Contrastive loss (diagonal should be highest)
        batch_size = similarities.size(0)
        labels = torch.arange(batch_size, device=similarities.device)
        
        loss = F.cross_entropy(similarities, labels)
        return loss
    
    def _evaluate(self, datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Evaluate model on all tasks"""
        self.model.eval()
        eval_metrics = {}
        
        with torch.no_grad():
            for task_name, task_dataset in datasets.items():
                # Sample small evaluation set
                eval_size = min(50, len(task_dataset))
                eval_samples = random.sample(task_dataset, eval_size)
                
                if task_name.endswith("_generation"):
                    metric = self._evaluate_generation(eval_samples)
                elif task_name.endswith("_retrieval"):
                    metric = self._evaluate_retrieval(eval_samples)
                elif task_name.endswith("_classification"):
                    metric = self._evaluate_classification(eval_samples, task_name)
                else:
                    metric = 0.0
                
                eval_metrics[f"{task_name}_metric"] = metric
        
        self.model.train()
        return eval_metrics
    
    def _evaluate_generation(self, samples: List[Dict[str, Any]]) -> float:
        """Evaluate generation task (placeholder)"""
        # Simple evaluation: compute average loss
        total_loss = 0.0
        tokenizer = getattr(self.model.base_model, 'tokenizer', None)
        
        if tokenizer is None:
            return 0.0
        
        for sample in samples[:10]:  # Evaluate on subset
            input_text = sample.get("input_text", "")
            target_text = sample.get("target_text", "")
            full_text = input_text + " " + target_text
            
            tokens = tokenizer(
                full_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model.base_model(**tokens, labels=tokens["input_ids"])
            total_loss += outputs.loss.item()
        
        return -total_loss / len(samples[:10])  # Negative loss as metric
    
    def _evaluate_retrieval(self, samples: List[Dict[str, Any]]) -> float:
        """Evaluate retrieval task"""
        # Compute average similarity between queries and positive docs
        total_similarity = 0.0
        tokenizer = getattr(self.model.base_model, 'tokenizer', None)
        
        if tokenizer is None:
            return 0.0
        
        for sample in samples[:10]:
            query_text = sample.get("query_text", "")
            positive_doc = sample.get("positive_doc", "")
            
            # Tokenize
            query_tokens = tokenizer(
                query_text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            doc_tokens = tokenizer(
                positive_doc, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            # Encode
            query_emb = self._encode_texts(
                query_tokens["input_ids"], query_tokens["attention_mask"]
            )
            doc_emb = self._encode_texts(
                doc_tokens["input_ids"], doc_tokens["attention_mask"]
            )
            
            # Similarity
            similarity = F.cosine_similarity(query_emb, doc_emb, dim=1)
            total_similarity += similarity.item()
        
        return total_similarity / len(samples[:10])
    
    def _evaluate_classification(self, samples: List[Dict[str, Any]], task_name: str) -> float:
        """Evaluate classification task"""
        correct = 0
        total = 0
        tokenizer = getattr(self.model.base_model, 'tokenizer', None)
        
        if tokenizer is None or task_name not in self.task_heads:
            return 0.0
        
        for sample in samples[:10]:
            text = sample.get("text", "")
            true_label = sample.get("label", 0)
            
            tokens = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            embeddings = self._encode_texts(
                tokens["input_ids"], tokens["attention_mask"]
            )
            
            logits = self.task_heads[task_name](embeddings)
            predicted = torch.argmax(logits, dim=1).item()
            
            if predicted == true_label:
                correct += 1
            total += 1
        
        return correct / max(total, 1)
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for multi-task data loader"""
        collated = {}
        
        # Get all keys from batch items
        all_keys = set()
        for item in batch:
            all_keys.update(item.keys())
        
        for key in all_keys:
            values = []
            for item in batch:
                if key in item:
                    values.append(item[key])
                else:
                    # Handle missing keys (pad with None or default values)
                    if key == "retrieval_context":
                        values.append(None)
                    elif key.endswith("_ids") or key.endswith("_mask"):
                        # Create dummy tensor for missing text fields
                        values.append(torch.zeros(512, dtype=torch.long))
                    else:
                        values.append(None)
            
            # Stack tensors, keep lists for non-tensors
            if values and isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        
        return collated
    
    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps: int, num_training_steps: int):
        """Create linear learning rate schedule with warmup"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, 
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _save_checkpoint(self, checkpoint_path: str, global_step: int):
        """Save training checkpoint"""
        import os
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save adapter
        self.model.save_adapter(os.path.join(checkpoint_path, "adapter.pt"))
        
        # Save task heads
        torch.save(
            self.task_heads.state_dict(),
            os.path.join(checkpoint_path, "task_heads.pt")
        )
        
        # Save training state
        training_state = {
            "global_step": global_step,
            "tasks": self.tasks,
            "task_sampling": self.task_sampling,
            "temperature": self.temperature,
            "model_config": {
                "retrieval_dim": self.model.retrieval_dim,
                "fusion_method": self.model.fusion_method,
                "retrieval_weight": self.model.retrieval_weight
            }
        }
        
        torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))
        print(f"Multi-task checkpoint saved to: {checkpoint_path}")