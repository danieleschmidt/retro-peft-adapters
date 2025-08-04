"""
Contrastive retrieval training for improving retrieval-adapter alignment.

Implements contrastive learning to train adapters to better utilize
retrieved context through positive/negative sampling and alignment objectives.
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from ..adapters.base_adapter import BaseRetroAdapter


class ContrastiveRetrievalDataset(Dataset):
    """
    Dataset for contrastive retrieval training.
    
    Creates positive and negative retrieval examples for training
    adapters to better utilize retrieved context.
    """
    
    def __init__(
        self,
        queries: List[str],
        documents: List[str],
        query_doc_pairs: List[Tuple[int, int]],
        tokenizer,
        max_length: int = 512,
        negative_sampling_ratio: float = 2.0,
        hard_negative_ratio: float = 0.5
    ):
        """
        Initialize contrastive dataset.
        
        Args:
            queries: List of query texts
            documents: List of document texts
            query_doc_pairs: List of (query_idx, doc_idx) positive pairs
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            negative_sampling_ratio: Ratio of negatives to positives
            hard_negative_ratio: Ratio of hard negatives (vs random)
        """
        self.queries = queries
        self.documents = documents
        self.query_doc_pairs = query_doc_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_sampling_ratio = negative_sampling_ratio
        self.hard_negative_ratio = hard_negative_ratio
        
        # Build positive pairs mapping
        self.positive_pairs = {}
        for query_idx, doc_idx in query_doc_pairs:
            if query_idx not in self.positive_pairs:
                self.positive_pairs[query_idx] = []
            self.positive_pairs[query_idx].append(doc_idx)
        
        # Pre-compute document embeddings for hard negative mining
        self.document_embeddings = None  # Will be set by trainer
    
    def __len__(self):
        return len(self.query_doc_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_idx, positive_doc_idx = self.query_doc_pairs[idx]
        
        # Get positive example
        query_text = self.queries[query_idx]
        positive_doc = self.documents[positive_doc_idx]
        
        # Sample negative documents
        negative_docs = self._sample_negative_documents(
            query_idx, positive_doc_idx
        )
        
        # Tokenize texts
        query_tokens = self.tokenizer(
            query_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        positive_tokens = self.tokenizer(
            positive_doc,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        negative_tokens_list = []
        for neg_doc in negative_docs:
            neg_tokens = self.tokenizer(
                neg_doc,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            negative_tokens_list.append(neg_tokens)
        
        return {
            "query_input_ids": query_tokens["input_ids"].squeeze(0),
            "query_attention_mask": query_tokens["attention_mask"].squeeze(0),
            "positive_input_ids": positive_tokens["input_ids"].squeeze(0),
            "positive_attention_mask": positive_tokens["attention_mask"].squeeze(0),
            "negative_input_ids": torch.stack([
                neg_tokens["input_ids"].squeeze(0) 
                for neg_tokens in negative_tokens_list
            ]),
            "negative_attention_mask": torch.stack([
                neg_tokens["attention_mask"].squeeze(0)
                for neg_tokens in negative_tokens_list
            ]),
            "query_idx": torch.tensor(query_idx),
            "positive_doc_idx": torch.tensor(positive_doc_idx)
        }
    
    def _sample_negative_documents(
        self, 
        query_idx: int, 
        positive_doc_idx: int
    ) -> List[str]:
        """Sample negative documents for contrastive learning"""
        num_negatives = int(self.negative_sampling_ratio)
        num_hard_negatives = int(num_negatives * self.hard_negative_ratio)
        num_random_negatives = num_negatives - num_hard_negatives
        
        negative_docs = []
        
        # Get positive documents for this query (to avoid sampling them)
        positive_doc_indices = set(self.positive_pairs.get(query_idx, []))
        
        # Sample hard negatives (if embeddings available)
        if (self.document_embeddings is not None and 
            num_hard_negatives > 0 and 
            positive_doc_idx < len(self.document_embeddings)):
            
            hard_negatives = self._sample_hard_negatives(
                positive_doc_idx, num_hard_negatives, positive_doc_indices
            )
            negative_docs.extend([self.documents[idx] for idx in hard_negatives])
        
        # Sample random negatives
        remaining_negatives = num_negatives - len(negative_docs)
        if remaining_negatives > 0:
            available_indices = [
                i for i in range(len(self.documents))
                if i not in positive_doc_indices and i != positive_doc_idx
            ]
            
            random_negatives = random.sample(
                available_indices, 
                min(remaining_negatives, len(available_indices))
            )
            negative_docs.extend([self.documents[idx] for idx in random_negatives])
        
        # Pad with random documents if needed
        while len(negative_docs) < num_negatives:
            random_idx = random.randint(0, len(self.documents) - 1)
            if random_idx not in positive_doc_indices:
                negative_docs.append(self.documents[random_idx])
        
        return negative_docs[:num_negatives]
    
    def _sample_hard_negatives(
        self,
        positive_doc_idx: int,
        num_hard_negatives: int,
        exclude_indices: set
    ) -> List[int]:
        """Sample hard negatives based on embedding similarity"""
        positive_embedding = self.document_embeddings[positive_doc_idx]
        
        # Compute similarities
        similarities = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            if i not in exclude_indices and i != positive_doc_idx:
                similarity = F.cosine_similarity(
                    positive_embedding.unsqueeze(0),
                    doc_embedding.unsqueeze(0)
                ).item()
                similarities.append((i, similarity))
        
        # Sort by similarity (descending) and take top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Sample from top similar documents (hard negatives)
        top_candidates = min(len(similarities), num_hard_negatives * 3)
        hard_negative_pool = similarities[:top_candidates]
        
        # Randomly sample from top candidates
        selected_negatives = random.sample(
            hard_negative_pool,
            min(num_hard_negatives, len(hard_negative_pool))
        )
        
        return [idx for idx, sim in selected_negatives]
    
    def set_document_embeddings(self, embeddings: torch.Tensor):
        """Set document embeddings for hard negative mining"""
        self.document_embeddings = embeddings


class ContrastiveRetrievalTrainer:
    """
    Trainer for contrastive retrieval learning.
    
    Trains adapters to better utilize retrieved context through
    contrastive learning objectives.
    """
    
    def __init__(
        self,
        model: BaseRetroAdapter,
        temperature: float = 0.07,
        in_batch_negatives: bool = True,
        loss_type: str = "contrastive",  # "contrastive", "triplet", "circle"
        margin: float = 0.3,
        gamma: float = 80.0,  # Circle loss parameter
        alpha: float = 0.25   # Circle loss parameter
    ):
        """
        Initialize contrastive trainer.
        
        Args:
            model: Retrieval-augmented adapter to train
            temperature: Temperature for contrastive loss
            in_batch_negatives: Whether to use in-batch negatives
            loss_type: Type of contrastive loss
            margin: Margin for triplet loss
            gamma: Gamma parameter for circle loss
            alpha: Alpha parameter for circle loss
        """
        self.model = model
        self.temperature = temperature
        self.in_batch_negatives = in_batch_negatives
        self.loss_type = loss_type
        self.margin = margin
        self.gamma = gamma
        self.alpha = alpha
        
        # Setup device
        self.device = next(model.parameters()).device
    
    def train(
        self,
        dataset: ContrastiveRetrievalDataset,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        hard_negative_mining: bool = True,
        dynamic_hard_negatives: bool = True,
        curriculum_learning: bool = True,
        eval_steps: int = 500,
        save_steps: int = 1000,
        output_dir: str = "./contrastive_checkpoints"
    ):
        """
        Train with contrastive retrieval learning.
        
        Args:
            dataset: Contrastive dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate
            hard_negative_mining: Whether to use hard negative mining
            dynamic_hard_negatives: Whether to update hard negatives during training
            curriculum_learning: Whether to use curriculum learning
            eval_steps: Steps between evaluations
            save_steps: Steps between saves
            output_dir: Output directory for checkpoints
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * num_epochs
        
        scheduler = self._get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        
        # Pre-compute document embeddings for hard negative mining
        if hard_negative_mining:
            print("Computing document embeddings for hard negative mining...")
            doc_embeddings = self._compute_document_embeddings(dataset)
            dataset.set_document_embeddings(doc_embeddings)
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                loss_dict = self._compute_contrastive_loss(batch)
                loss = loss_dict["total_loss"]
                
                # Curriculum learning: gradually increase difficulty
                if curriculum_learning:
                    curriculum_weight = min(1.0, global_step / (total_steps * 0.3))
                    loss = curriculum_weight * loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % 100 == 0:
                    avg_loss = total_loss / global_step
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch}, Step {global_step}: "
                          f"Loss={loss.item():.4f}, "
                          f"Avg Loss={avg_loss:.4f}, "
                          f"LR={current_lr:.2e}")
                
                # Dynamic hard negative mining
                if (dynamic_hard_negatives and 
                    hard_negative_mining and 
                    global_step % 1000 == 0):
                    print("Updating document embeddings for hard negatives...")
                    doc_embeddings = self._compute_document_embeddings(dataset)
                    dataset.set_document_embeddings(doc_embeddings)
                
                # Evaluation
                if global_step % eval_steps == 0:
                    eval_metrics = self._evaluate(dataset)
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
    
    def _compute_contrastive_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss for batch"""
        batch_size = batch["query_input_ids"].size(0)
        
        # Encode queries
        query_embeddings = self._encode_texts(
            batch["query_input_ids"],
            batch["query_attention_mask"]
        )
        
        # Encode positive documents
        positive_embeddings = self._encode_texts(
            batch["positive_input_ids"],
            batch["positive_attention_mask"]
        )
        
        # Encode negative documents
        num_negatives = batch["negative_input_ids"].size(1)
        negative_input_ids = batch["negative_input_ids"].view(-1, batch["negative_input_ids"].size(-1))
        negative_attention_mask = batch["negative_attention_mask"].view(-1, batch["negative_attention_mask"].size(-1))
        
        negative_embeddings = self._encode_texts(negative_input_ids, negative_attention_mask)
        negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)
        
        # Compute loss based on loss type
        if self.loss_type == "contrastive":
            loss = self._contrastive_loss(
                query_embeddings, positive_embeddings, negative_embeddings
            )
        elif self.loss_type == "triplet":
            loss = self._triplet_loss(
                query_embeddings, positive_embeddings, negative_embeddings
            )
        elif self.loss_type == "circle":
            loss = self._circle_loss(
                query_embeddings, positive_embeddings, negative_embeddings
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return {
            "total_loss": loss,
            "contrastive_loss": loss
        }
    
    def _encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode texts using model encoder"""
        # Get embeddings from base model
        embeddings = self.model.base_model.get_input_embeddings()(input_ids)
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_embeddings = sum_embeddings / sum_mask
        
        # Project to retrieval dimension
        projected_embeddings = self.model.retrieval_projector(pooled_embeddings)
        
        return F.normalize(projected_embeddings, p=2, dim=1)
    
    def _contrastive_loss(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss"""
        batch_size = query_embeddings.size(0)
        
        # Positive similarities
        positive_similarities = F.cosine_similarity(
            query_embeddings, positive_embeddings, dim=1
        ) / self.temperature
        
        # Negative similarities
        negative_similarities = torch.bmm(
            query_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1) / self.temperature
        
        # In-batch negatives (optional)
        if self.in_batch_negatives:
            # Use other positives in batch as additional negatives
            all_positives = positive_embeddings
            in_batch_similarities = torch.mm(
                query_embeddings, all_positives.t()
            ) / self.temperature
            
            # Mask out self-similarities
            mask = torch.eye(batch_size, device=query_embeddings.device).bool()
            in_batch_similarities = in_batch_similarities.masked_fill(mask, float('-inf'))
            
            # Combine with explicit negatives
            all_negative_similarities = torch.cat([
                negative_similarities, in_batch_similarities
            ], dim=1)
        else:
            all_negative_similarities = negative_similarities
        
        # Contrastive loss
        logits = torch.cat([
            positive_similarities.unsqueeze(1),
            all_negative_similarities
        ], dim=1)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def _triplet_loss(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute triplet loss"""
        # Positive distances
        positive_distances = 1 - F.cosine_similarity(
            query_embeddings, positive_embeddings, dim=1
        )
        
        # Negative distances (take minimum for hardest negative)
        negative_similarities = torch.bmm(
            query_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1)
        
        negative_distances = 1 - negative_similarities
        hardest_negative_distances = negative_distances.min(dim=1)[0]
        
        # Triplet loss
        loss = F.relu(positive_distances - hardest_negative_distances + self.margin)
        return loss.mean()
    
    def _circle_loss(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute circle loss"""
        # Positive similarities
        positive_similarities = F.cosine_similarity(
            query_embeddings, positive_embeddings, dim=1
        )
        
        # Negative similarities
        negative_similarities = torch.bmm(
            query_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1)
        
        # Circle loss computation
        alpha_p = torch.clamp_min(self.alpha + 1 - positive_similarities.detach(), min=0.0)
        alpha_n = torch.clamp_min(negative_similarities.detach() + self.alpha, min=0.0)
        
        delta_p = 1 - self.alpha
        delta_n = self.alpha
        
        # Positive term
        positive_term = alpha_p * (positive_similarities - delta_p)
        
        # Negative term  
        negative_term = alpha_n * (negative_similarities - delta_n)
        
        # Circle loss
        loss = torch.log(
            1 + torch.sum(torch.exp(self.gamma * negative_term), dim=1) * 
            torch.exp(-self.gamma * positive_term)
        )
        
        return loss.mean()
    
    def _compute_document_embeddings(self, dataset: ContrastiveRetrievalDataset) -> torch.Tensor:
        """Compute embeddings for all documents (for hard negative mining)"""
        self.model.eval()
        
        doc_embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(dataset.documents), batch_size):
                batch_docs = dataset.documents[i:i + batch_size]
                
                # Tokenize batch
                tokens = dataset.tokenizer(
                    batch_docs,
                    max_length=dataset.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                # Encode
                embeddings = self._encode_texts(
                    tokens["input_ids"],
                    tokens["attention_mask"]
                )
                
                doc_embeddings.append(embeddings.cpu())
        
        self.model.train()
        return torch.cat(doc_embeddings, dim=0)
    
    def _evaluate(self, dataset: ContrastiveRetrievalDataset) -> Dict[str, float]:
        """Evaluate model on retrieval metrics"""
        # Simple evaluation: compute average similarity between queries and positives
        self.model.eval()
        
        total_similarity = 0.0
        num_samples = min(100, len(dataset))  # Sample for efficiency
        
        with torch.no_grad():
            for i in range(0, num_samples, 10):
                batch_end = min(i + 10, num_samples)
                batch_data = [dataset[j] for j in range(i, batch_end)]
                
                # Compute similarities
                for data in batch_data:
                    query_emb = self._encode_texts(
                        data["query_input_ids"].unsqueeze(0).to(self.device),
                        data["query_attention_mask"].unsqueeze(0).to(self.device)
                    )
                    
                    positive_emb = self._encode_texts(
                        data["positive_input_ids"].unsqueeze(0).to(self.device),
                        data["positive_attention_mask"].unsqueeze(0).to(self.device)
                    )
                    
                    similarity = F.cosine_similarity(query_emb, positive_emb, dim=1)
                    total_similarity += similarity.item()
        
        self.model.train()
        
        return {
            "avg_positive_similarity": total_similarity / num_samples
        }
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for data loader"""
        collated = {}
        
        for key in batch[0].keys():
            if key in ["negative_input_ids", "negative_attention_mask"]:
                # Stack negative examples
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                # Regular stacking
                collated[key] = torch.stack([item[key] for item in batch])
        
        return collated
    
    def _get_linear_schedule_with_warmup(
        self, 
        optimizer, 
        num_warmup_steps: int, 
        num_training_steps: int
    ):
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
        
        # Save training state
        training_state = {
            "global_step": global_step,
            "temperature": self.temperature,
            "loss_type": self.loss_type,
            "model_config": {
                "retrieval_dim": self.model.retrieval_dim,
                "fusion_method": self.model.fusion_method,
                "retrieval_weight": self.model.retrieval_weight
            }
        }
        
        torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))
        print(f"Checkpoint saved to: {checkpoint_path}")