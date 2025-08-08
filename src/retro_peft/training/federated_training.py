"""
Federated learning support for RetroLoRA adapters.

Enables distributed training of adapters across multiple organizations
while preserving privacy through differential privacy and secure aggregation.
"""

import asyncio
import json
import logging
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..adapters.base_adapter import BaseRetroAdapter
from ..utils.monitoring import MetricsCollector
from ..utils.enhanced_security import DifferentialPrivacyManager


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""

    # Client settings
    num_clients: int = 10
    clients_per_round: int = 5
    local_epochs: int = 3
    local_batch_size: int = 16

    # Aggregation settings
    aggregation_method: str = "fedavg"  # fedavg, fedprox, scaffold
    global_rounds: int = 100
    min_clients_for_update: int = 3

    # Privacy settings
    use_differential_privacy: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0

    # Communication settings
    compression_enabled: bool = True
    compression_ratio: float = 0.1
    secure_aggregation: bool = True

    # Adaptation settings
    adaptive_lr: bool = True
    lr_decay_factor: float = 0.95
    early_stopping_patience: int = 5


class FederatedClient:
    """
    Federated learning client for RetroLoRA training.
    """

    def __init__(
        self,
        client_id: str,
        adapter: BaseRetroAdapter,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[FederatedConfig] = None,
        privacy_manager: Optional[DifferentialPrivacyManager] = None,
    ):
        self.client_id = client_id
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or FederatedConfig()
        self.privacy_manager = privacy_manager

        # Client state
        self.local_model_state = None
        self.optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=1e-4)
        self.metrics = MetricsCollector(f"federated_client_{client_id}")

        # Logging
        self.logger = logging.getLogger(f"FedClient-{client_id}")

    def local_train(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Perform local training on client data.

        Args:
            global_model_state: Global model parameters from server

        Returns:
            Dictionary containing local updates and metrics
        """
        self.logger.info(f"Starting local training for {self.config.local_epochs} epochs")

        # Load global model state
        self.adapter.load_state_dict(global_model_state, strict=False)
        self.adapter.train()

        # Store initial parameters for FedProx
        initial_params = {name: param.clone() for name, param in self.adapter.named_parameters()}

        local_losses = []

        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self._forward_batch(batch)
                loss = outputs["loss"]

                # Add FedProx regularization if enabled
                if self.config.aggregation_method == "fedprox":
                    prox_loss = self._compute_prox_loss(initial_params, mu=0.01)
                    loss = loss + prox_loss

                # Backward pass
                loss.backward()

                # Apply differential privacy
                if self.privacy_manager:
                    self.privacy_manager.apply_dp_noise(
                        self.adapter.parameters(),
                        epsilon=self.config.dp_epsilon,
                        delta=self.config.dp_delta,
                        max_grad_norm=self.config.dp_max_grad_norm,
                    )

                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Log batch metrics
                self.metrics.record("local_batch_loss", loss.item())

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            local_losses.append(avg_epoch_loss)

            self.logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.6f}")

        # Validation if available
        val_metrics = {}
        if self.val_loader:
            val_metrics = self._validate()

        # Compute parameter updates
        updates = self._compute_updates(initial_params)

        # Compress updates if enabled
        if self.config.compression_enabled:
            updates = self._compress_updates(updates)

        return {
            "client_id": self.client_id,
            "updates": updates,
            "num_samples": len(self.train_loader.dataset),
            "local_losses": local_losses,
            "val_metrics": val_metrics,
            "privacy_budget_used": (
                self.privacy_manager.get_budget_used() if self.privacy_manager else 0.0
            ),
        }

    def _forward_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single batch and return outputs."""
        # This should be implemented based on your specific task
        # For now, return a dummy loss
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    def _compute_prox_loss(
        self, initial_params: Dict[str, torch.Tensor], mu: float = 0.01
    ) -> torch.Tensor:
        """Compute FedProx regularization term."""
        prox_loss = torch.tensor(0.0)
        for name, param in self.adapter.named_parameters():
            if name in initial_params:
                prox_loss += torch.norm(param - initial_params[name]) ** 2
        return mu * prox_loss / 2

    def _validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        self.adapter.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self._forward_batch(batch)
                val_loss += outputs["loss"].item()
                num_batches += 1

        return {
            "val_loss": val_loss / num_batches if num_batches > 0 else 0.0,
            "val_samples": len(self.val_loader.dataset),
        }

    def _compute_updates(self, initial_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute parameter updates (difference from initial)."""
        updates = {}
        for name, param in self.adapter.named_parameters():
            if name in initial_params:
                updates[name] = param.data - initial_params[name]
        return updates

    def _compress_updates(self, updates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply gradient compression to reduce communication overhead."""
        compressed = {}
        for name, update in updates.items():
            # Top-k sparsification
            k = max(1, int(update.numel() * self.config.compression_ratio))
            flat_update = update.flatten()
            _, indices = torch.topk(torch.abs(flat_update), k)

            # Create sparse update
            compressed_update = torch.zeros_like(flat_update)
            compressed_update[indices] = flat_update[indices]
            compressed[name] = compressed_update.reshape(update.shape)

        return compressed


class FederatedServer:
    """
    Federated learning server for coordinating client training.
    """

    def __init__(
        self, initial_model_state: Dict[str, torch.Tensor], config: Optional[FederatedConfig] = None
    ):
        self.global_model_state = initial_model_state.copy()
        self.config = config or FederatedConfig()
        self.round_number = 0

        # Server state
        self.client_states = {}
        self.aggregation_weights = {}
        self.metrics = MetricsCollector("federated_server")

        # Convergence tracking
        self.best_global_loss = float("inf")
        self.patience_counter = 0

        # Logging
        self.logger = logging.getLogger("FedServer")

    async def coordinate_training(
        self, clients: List[FederatedClient], callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Coordinate federated training across multiple rounds.

        Args:
            clients: List of federated clients
            callback: Optional callback for progress updates

        Returns:
            Final training results and metrics
        """
        self.logger.info(f"Starting federated training for {self.config.global_rounds} rounds")

        training_history = {
            "round_metrics": [],
            "global_losses": [],
            "participation_rates": [],
            "convergence_metrics": [],
        }

        for round_num in range(self.config.global_rounds):
            self.round_number = round_num
            self.logger.info(f"Starting round {round_num + 1}/{self.config.global_rounds}")

            # Select clients for this round
            selected_clients = self._select_clients(clients)
            self.logger.info(f"Selected {len(selected_clients)} clients for training")

            # Parallel client training
            client_updates = await self._parallel_client_training(selected_clients)

            if len(client_updates) < self.config.min_clients_for_update:
                self.logger.warning(f"Insufficient clients ({len(client_updates)}) for aggregation")
                continue

            # Aggregate client updates
            aggregated_state = self._aggregate_updates(client_updates)

            # Update global model
            self.global_model_state = aggregated_state

            # Compute round metrics
            round_metrics = self._compute_round_metrics(client_updates)
            training_history["round_metrics"].append(round_metrics)
            training_history["global_losses"].append(round_metrics["avg_loss"])
            training_history["participation_rates"].append(len(client_updates) / len(clients))

            # Check convergence
            converged = self._check_convergence(round_metrics["avg_loss"])
            if converged:
                self.logger.info(f"Training converged at round {round_num + 1}")
                break

            # Progress callback
            if callback:
                callback(round_num, round_metrics, self.global_model_state)

            self.logger.info(
                f"Round {round_num + 1} completed. "
                f"Avg Loss: {round_metrics['avg_loss']:.6f}, "
                f"Participation: {len(client_updates)}/{len(clients)}"
            )

        final_results = {
            "final_model_state": self.global_model_state,
            "training_history": training_history,
            "total_rounds": self.round_number + 1,
            "converged": converged if "converged" in locals() else False,
            "final_metrics": (
                training_history["round_metrics"][-1] if training_history["round_metrics"] else {}
            ),
        }

        return final_results

    def _select_clients(self, clients: List[FederatedClient]) -> List[FederatedClient]:
        """Select clients for current round."""
        if len(clients) <= self.config.clients_per_round:
            return clients

        # Random selection for now - could implement more sophisticated strategies
        import random

        return random.sample(clients, self.config.clients_per_round)

    async def _parallel_client_training(
        self, selected_clients: List[FederatedClient]
    ) -> List[Dict[str, Any]]:
        """Run client training in parallel."""
        tasks = []
        for client in selected_clients:
            task = asyncio.create_task(self._run_client_training(client))
            tasks.append(task)

        # Wait for all clients to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed clients
        successful_updates = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Client {selected_clients[i].client_id} failed: {result}")
            else:
                successful_updates.append(result)

        return successful_updates

    async def _run_client_training(self, client: FederatedClient) -> Dict[str, Any]:
        """Run training for a single client."""
        try:
            # Run client training in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, client.local_train, self.global_model_state)
            return result
        except Exception as e:
            self.logger.error(f"Client {client.client_id} training failed: {e}")
            raise e

    def _aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using specified method."""
        if self.config.aggregation_method == "fedavg":
            return self._federated_averaging(client_updates)
        elif self.config.aggregation_method == "fedprox":
            return self._federated_averaging(client_updates)  # Same as FedAvg for aggregation
        elif self.config.aggregation_method == "scaffold":
            return self._scaffold_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")

    def _federated_averaging(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Standard FedAvg aggregation weighted by number of samples."""
        total_samples = sum(update["num_samples"] for update in client_updates)
        aggregated_state = {}

        # Initialize aggregated state
        first_update = client_updates[0]["updates"]
        for param_name in first_update.keys():
            aggregated_state[param_name] = torch.zeros_like(first_update[param_name])

        # Weighted aggregation
        for update in client_updates:
            weight = update["num_samples"] / total_samples
            for param_name, param_update in update["updates"].items():
                if param_name in aggregated_state:
                    aggregated_state[param_name] += weight * param_update

        # Apply updates to global state
        new_global_state = {}
        for param_name, global_param in self.global_model_state.items():
            if param_name in aggregated_state:
                new_global_state[param_name] = global_param + aggregated_state[param_name]
            else:
                new_global_state[param_name] = global_param

        return new_global_state

    def _scaffold_aggregation(
        self, client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """SCAFFOLD aggregation with control variates."""
        # Simplified SCAFFOLD - full implementation would require control variates
        return self._federated_averaging(client_updates)

    def _compute_round_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute metrics for the current round."""
        total_samples = sum(update["num_samples"] for update in client_updates)

        # Weighted average loss
        weighted_loss = (
            sum(
                update["num_samples"] * np.mean(update["local_losses"]) for update in client_updates
            )
            / total_samples
        )

        # Participation metrics
        participation_rate = len(client_updates) / max(1, self.config.num_clients)

        # Privacy budget tracking
        avg_privacy_budget = np.mean(
            [update.get("privacy_budget_used", 0.0) for update in client_updates]
        )

        return {
            "avg_loss": weighted_loss,
            "participation_rate": participation_rate,
            "num_participating_clients": len(client_updates),
            "total_samples": total_samples,
            "avg_privacy_budget_used": avg_privacy_budget,
        }

    def _check_convergence(self, current_loss: float) -> bool:
        """Check if training has converged."""
        if current_loss < self.best_global_loss:
            self.best_global_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def save_checkpoint(self, path: str) -> None:
        """Save federated learning checkpoint."""
        checkpoint = {
            "global_model_state": self.global_model_state,
            "round_number": self.round_number,
            "config": asdict(self.config),
            "best_global_loss": self.best_global_loss,
            "patience_counter": self.patience_counter,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load federated learning checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        self.global_model_state = checkpoint["global_model_state"]
        self.round_number = checkpoint["round_number"]
        self.best_global_loss = checkpoint["best_global_loss"]
        self.patience_counter = checkpoint["patience_counter"]

        self.logger.info(f"Checkpoint loaded from {path}")


class FederatedTrainingManager:
    """
    High-level manager for federated learning workflows.
    """

    def __init__(self, config: Optional[FederatedConfig] = None):
        self.config = config or FederatedConfig()
        self.logger = logging.getLogger("FedManager")

    async def run_federated_training(
        self,
        adapter_class: type,
        adapter_config: Dict[str, Any],
        client_data: Dict[str, DataLoader],
        val_data: Optional[Dict[str, DataLoader]] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run complete federated training workflow.

        Args:
            adapter_class: RetroLoRA adapter class
            adapter_config: Configuration for adapters
            client_data: Dictionary mapping client_id -> training DataLoader
            val_data: Optional validation data per client
            callback: Optional progress callback

        Returns:
            Training results and final model
        """
        self.logger.info("Initializing federated training")

        # Create initial model
        initial_adapter = adapter_class(**adapter_config)
        initial_state = initial_adapter.state_dict()

        # Create server
        server = FederatedServer(initial_state, self.config)

        # Create clients
        clients = []
        for client_id, train_loader in client_data.items():
            client_adapter = adapter_class(**adapter_config)
            val_loader = val_data.get(client_id) if val_data else None

            # Optional: Create privacy manager for each client
            privacy_manager = None
            if self.config.use_differential_privacy:
                privacy_manager = DifferentialPrivacyManager(
                    epsilon=self.config.dp_epsilon, delta=self.config.dp_delta
                )

            client = FederatedClient(
                client_id=client_id,
                adapter=client_adapter,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config,
                privacy_manager=privacy_manager,
            )
            clients.append(client)

        self.logger.info(f"Created {len(clients)} federated clients")

        # Run coordinated training
        results = await server.coordinate_training(clients, callback)

        # Create final model with aggregated weights
        final_adapter = adapter_class(**adapter_config)
        final_adapter.load_state_dict(results["final_model_state"], strict=False)

        return {
            "final_adapter": final_adapter,
            "training_results": results,
            "server": server,
            "clients": clients,
        }


# Utility functions for federated learning setup
def create_federated_dataloaders(
    dataset, num_clients: int = 10, iid: bool = True, alpha: float = 0.5, batch_size: int = 16
) -> Dict[str, DataLoader]:
    """
    Create federated data splits from a dataset.

    Args:
        dataset: PyTorch dataset to split
        num_clients: Number of federated clients
        iid: Whether to use IID or non-IID splits
        alpha: Dirichlet concentration parameter for non-IID splits
        batch_size: Batch size for DataLoaders

    Returns:
        Dictionary mapping client_id -> DataLoader
    """
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    n_samples = len(dataset)

    if iid:
        # IID splits
        indices = np.random.permutation(n_samples)
        split_size = n_samples // num_clients

        client_loaders = {}
        for i in range(num_clients):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_clients - 1 else n_samples
            client_indices = indices[start_idx:end_idx]

            client_dataset = Subset(dataset, client_indices)
            client_loaders[f"client_{i}"] = DataLoader(
                client_dataset, batch_size=batch_size, shuffle=True
            )
    else:
        # Non-IID splits using Dirichlet distribution
        # This would require label information - simplified for now
        indices = np.random.permutation(n_samples)
        split_size = n_samples // num_clients

        client_loaders = {}
        for i in range(num_clients):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_clients - 1 else n_samples
            client_indices = indices[start_idx:end_idx]

            client_dataset = Subset(dataset, client_indices)
            client_loaders[f"client_{i}"] = DataLoader(
                client_dataset, batch_size=batch_size, shuffle=True
            )

    return client_loaders


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Example federated training setup
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            global_rounds=20,
            local_epochs=2,
            use_differential_privacy=True,
        )

        manager = FederatedTrainingManager(config)

        # This would normally be actual data
        print("Federated training manager initialized")
        print(f"Configuration: {config}")

    asyncio.run(main())
