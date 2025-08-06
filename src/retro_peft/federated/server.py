"""
Federated Learning Server for Retro-PEFT-Adapters

Implements the server-side logic for federated learning with advanced features:
- Secure aggregation with differential privacy
- Adaptive client selection and weighting
- Asynchronous and synchronous training modes
- Byzantine fault tolerance
- Performance monitoring and analytics
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from .aggregation import FederatedAggregator, AggregationStrategy
from .privacy import PrivacyEngine, PrivacyConfig
from .communication import SecureCommunicator, MessageType

logger = logging.getLogger(__name__)


@dataclass
class FederatedServerConfig:
    """Configuration for federated learning server"""
    server_id: str = "federated_server"
    max_clients: int = 100
    min_clients_per_round: int = 2
    max_clients_per_round: int = 10
    client_selection_strategy: str = "random"  # random, performance_based, resource_aware
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox, scaffold, fednova
    
    # Training parameters
    global_rounds: int = 100
    client_epochs: int = 1
    client_batch_size: int = 32
    learning_rate: float = 1e-4
    
    # Privacy parameters
    enable_privacy: bool = True
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Communication parameters
    communication_rounds_timeout: float = 300.0  # 5 minutes
    use_compression: bool = True
    compression_ratio: float = 0.1
    
    # Byzantine tolerance
    enable_byzantine_tolerance: bool = True
    byzantine_threshold: float = 0.3
    anomaly_detection_threshold: float = 2.0
    
    # Asynchronous mode
    enable_async_mode: bool = False
    async_staleness_threshold: int = 5
    
    # Monitoring
    enable_performance_monitoring: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 10


@dataclass
class ClientInfo:
    """Information about a federated client"""
    client_id: str
    join_time: float
    last_seen: float
    data_size: int
    compute_capacity: float
    network_bandwidth: float
    privacy_budget_remaining: float
    performance_history: List[float]
    is_active: bool = True
    round_participation: int = 0
    byzantine_score: float = 0.0


@dataclass
class FederatedRound:
    """Information about a federated learning round"""
    round_id: int
    start_time: float
    end_time: Optional[float]
    selected_clients: List[str]
    client_updates: Dict[str, Any]
    aggregated_weights: Optional[Dict[str, torch.Tensor]]
    global_loss: Optional[float]
    global_accuracy: Optional[float]
    privacy_spent: float
    communication_cost: float


class FederatedServer:
    """
    Advanced federated learning server for retro-peft adapters
    
    Features:
    - Multi-strategy client selection and weighting
    - Privacy-preserving aggregation with differential privacy
    - Byzantine fault tolerance and anomaly detection
    - Asynchronous and synchronous training modes
    - Adaptive learning rate scheduling
    - Comprehensive monitoring and analytics
    """
    
    def __init__(
        self,
        config: FederatedServerConfig,
        model_config: Dict[str, Any],
        output_dir: str = "federated_server"
    ):
        self.config = config
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Server state
        self.server_id = config.server_id
        self.current_round = 0
        self.is_training = False
        self.training_start_time: Optional[float] = None
        
        # Client management
        self.clients: Dict[str, ClientInfo] = {}
        self.client_lock = threading.Lock()
        
        # Global model
        self.global_model = self._initialize_global_model()
        self.model_lock = threading.Lock()
        
        # Training history
        self.training_history: List[FederatedRound] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Components
        self.aggregator = FederatedAggregator(
            strategy=AggregationStrategy(config.aggregation_strategy),
            privacy_config=PrivacyConfig(
                epsilon=config.privacy_epsilon,
                delta=config.privacy_delta,
                max_grad_norm=config.max_grad_norm
            ) if config.enable_privacy else None
        )
        
        self.privacy_engine = PrivacyEngine(
            epsilon=config.privacy_epsilon,
            delta=config.privacy_delta,
            max_clients=config.max_clients
        ) if config.enable_privacy else None
        
        self.communicator = SecureCommunicator(
            node_id=self.server_id,
            use_encryption=True
        )
        
        # Async components
        if config.enable_async_mode:
            self.update_queue = asyncio.Queue()
            self.staleness_tracker: Dict[str, int] = {}
            
        # Thread pool for handling concurrent client communications
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_clients)
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Federated server initialized: {self.server_id}")
        logger.info(f"Configuration: {asdict(config)}")
        
    def _initialize_global_model(self) -> nn.Module:
        """Initialize the global model based on configuration"""
        # This would typically load a pre-trained model and add adapters
        # For demonstration, we'll use a simple placeholder
        
        from ..adapters import RetroLoRA
        
        # Mock base model dimensions
        input_dim = self.model_config.get("input_dim", 768)
        output_dim = self.model_config.get("output_dim", 768)
        
        # Create RetroLoRA adapter as global model
        global_model = RetroLoRA(
            base_model=None,  # Would be actual base model
            r=self.model_config.get("rank", 16),
            alpha=self.model_config.get("alpha", 32),
            target_modules=self.model_config.get("target_modules", ["q_proj", "v_proj"])
        )
        
        return global_model
        
    def _setup_logging(self):
        """Setup detailed logging for federated training"""
        log_file = self.output_dir / f"{self.server_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        
    def register_client(
        self,
        client_id: str,
        client_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Register a new client for federated learning
        
        Args:
            client_id: Unique identifier for the client
            client_info: Information about client capabilities
            
        Returns:
            Registration response with server information
        """
        with self.client_lock:
            if len(self.clients) >= self.config.max_clients:
                return {
                    "success": False,
                    "error": "Server at maximum client capacity"
                }
                
            if client_id in self.clients:
                # Update existing client info
                existing_client = self.clients[client_id]
                existing_client.last_seen = time.time()
                existing_client.is_active = True
                logger.info(f"Client {client_id} reconnected")
            else:
                # Register new client
                new_client = ClientInfo(
                    client_id=client_id,
                    join_time=time.time(),
                    last_seen=time.time(),
                    data_size=client_info.get("data_size", 0),
                    compute_capacity=client_info.get("compute_capacity", 1.0),
                    network_bandwidth=client_info.get("network_bandwidth", 1.0),
                    privacy_budget_remaining=self.config.privacy_epsilon if self.config.enable_privacy else float('inf'),
                    performance_history=[]
                )
                
                self.clients[client_id] = new_client
                logger.info(f"New client registered: {client_id}")
                
        # Prepare registration response
        response = {
            "success": True,
            "server_id": self.server_id,
            "global_round": self.current_round,
            "training_config": {
                "client_epochs": self.config.client_epochs,
                "batch_size": self.config.client_batch_size,
                "learning_rate": self.config.learning_rate
            },
            "privacy_config": {
                "enable_privacy": self.config.enable_privacy,
                "epsilon": self.config.privacy_epsilon,
                "delta": self.config.privacy_delta
            } if self.config.enable_privacy else None
        }
        
        return response
        
    def start_federated_training(self) -> None:
        """Start federated learning training process"""
        if self.is_training:
            logger.warning("Federated training already in progress")
            return
            
        logger.info("Starting federated learning training")
        self.is_training = True
        self.training_start_time = time.time()
        
        try:
            if self.config.enable_async_mode:
                asyncio.run(self._run_async_training())
            else:
                self._run_sync_training()
                
        except Exception as e:
            logger.error(f"Error during federated training: {str(e)}")
            raise
        finally:
            self.is_training = False
            
    def _run_sync_training(self) -> None:
        """Run synchronous federated training"""
        logger.info("Starting synchronous federated training")
        
        for round_id in range(self.current_round, self.config.global_rounds):
            logger.info(f"Starting federated round {round_id + 1}/{self.config.global_rounds}")
            
            # Create new round
            fed_round = FederatedRound(
                round_id=round_id,
                start_time=time.time(),
                end_time=None,
                selected_clients=[],
                client_updates={},
                aggregated_weights=None,
                global_loss=None,
                global_accuracy=None,
                privacy_spent=0.0,
                communication_cost=0.0
            )
            
            try:
                # Select clients for this round
                selected_clients = self._select_clients_for_round()
                fed_round.selected_clients = selected_clients
                
                if len(selected_clients) < self.config.min_clients_per_round:
                    logger.warning(f"Not enough clients available ({len(selected_clients)} < {self.config.min_clients_per_round})")
                    continue
                    
                logger.info(f"Selected {len(selected_clients)} clients: {selected_clients}")
                
                # Send global model to selected clients
                global_weights = self._get_global_weights()
                training_tasks = []
                
                for client_id in selected_clients:
                    task = self.thread_pool.submit(
                        self._send_training_request,
                        client_id,
                        global_weights,
                        round_id
                    )
                    training_tasks.append((client_id, task))
                    
                # Collect client updates
                client_updates = {}
                round_start = time.time()
                
                for client_id, task in training_tasks:
                    try:
                        # Wait for client response with timeout
                        client_update = task.result(timeout=self.config.communication_rounds_timeout)
                        if client_update is not None:
                            client_updates[client_id] = client_update
                            logger.debug(f"Received update from client {client_id}")
                        else:
                            logger.warning(f"No update received from client {client_id}")
                            
                    except Exception as e:
                        logger.error(f"Error getting update from client {client_id}: {str(e)}")
                        self._handle_client_failure(client_id)
                        
                fed_round.client_updates = client_updates
                
                if not client_updates:
                    logger.warning("No client updates received for this round")
                    continue
                    
                # Byzantine fault tolerance check
                if self.config.enable_byzantine_tolerance:
                    client_updates = self._filter_byzantine_updates(client_updates)
                    
                # Aggregate client updates
                logger.info(f"Aggregating updates from {len(client_updates)} clients")
                aggregated_weights, privacy_spent = self._aggregate_client_updates(
                    client_updates,
                    [self.clients[cid].data_size for cid in client_updates.keys()]
                )
                
                fed_round.aggregated_weights = aggregated_weights
                fed_round.privacy_spent = privacy_spent
                
                # Update global model
                with self.model_lock:
                    self._update_global_model(aggregated_weights)
                    
                # Evaluate global model (if evaluation data available)
                global_metrics = self._evaluate_global_model()
                fed_round.global_loss = global_metrics.get("loss")
                fed_round.global_accuracy = global_metrics.get("accuracy")
                
                # Update client performance histories
                self._update_client_histories(client_updates)
                
                # Record round completion
                fed_round.end_time = time.time()
                round_duration = fed_round.end_time - fed_round.start_time
                
                self.training_history.append(fed_round)
                self.current_round = round_id + 1
                
                # Log round results
                logger.info(f"Round {round_id + 1} completed in {round_duration:.2f}s")
                if fed_round.global_accuracy:
                    logger.info(f"Global accuracy: {fed_round.global_accuracy:.4f}")
                if fed_round.privacy_spent > 0:
                    logger.info(f"Privacy spent: {fed_round.privacy_spent:.6f}")
                    
                # Save checkpoint periodically
                if self.config.save_checkpoints and (round_id + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(round_id + 1)
                    
                # Performance monitoring
                if self.config.enable_performance_monitoring:
                    self._update_performance_metrics(fed_round)
                    
            except Exception as e:
                logger.error(f"Error in round {round_id}: {str(e)}")
                continue
                
        logger.info("Synchronous federated training completed")
        self._save_final_results()
        
    async def _run_async_training(self) -> None:
        """Run asynchronous federated training"""
        logger.info("Starting asynchronous federated training")
        
        # Start background task to process client updates
        update_processor = asyncio.create_task(self._process_async_updates())
        
        # Send initial training requests
        await self._broadcast_async_training_request()
        
        # Wait for training completion or timeout
        training_duration = 0
        max_training_time = self.config.global_rounds * 300  # 5 minutes per round equivalent
        
        while training_duration < max_training_time and self.current_round < self.config.global_rounds:
            await asyncio.sleep(10)  # Check every 10 seconds
            training_duration = time.time() - self.training_start_time
            
        # Clean up
        update_processor.cancel()
        logger.info("Asynchronous federated training completed")
        self._save_final_results()
        
    def _select_clients_for_round(self) -> List[str]:
        """Select clients for the current training round"""
        with self.client_lock:
            active_clients = [
                client_id for client_id, client_info in self.clients.items()
                if client_info.is_active and self._is_client_eligible(client_info)
            ]
            
        if not active_clients:
            return []
            
        num_selected = min(
            len(active_clients),
            self.config.max_clients_per_round
        )
        
        if self.config.client_selection_strategy == "random":
            selected = np.random.choice(active_clients, size=num_selected, replace=False).tolist()
            
        elif self.config.client_selection_strategy == "performance_based":
            # Select based on historical performance
            client_scores = []
            for client_id in active_clients:
                client_info = self.clients[client_id]
                # Score based on average performance and participation
                avg_performance = np.mean(client_info.performance_history) if client_info.performance_history else 0.5
                participation_bonus = min(client_info.round_participation / max(self.current_round, 1), 0.1)
                score = avg_performance + participation_bonus
                client_scores.append((client_id, score))
                
            # Select top performers
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [client_id for client_id, _ in client_scores[:num_selected]]
            
        elif self.config.client_selection_strategy == "resource_aware":
            # Select based on compute capacity and network bandwidth
            client_scores = []
            for client_id in active_clients:
                client_info = self.clients[client_id]
                resource_score = (client_info.compute_capacity * 0.6 + 
                                client_info.network_bandwidth * 0.4)
                client_scores.append((client_id, resource_score))
                
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [client_id for client_id, _ in client_scores[:num_selected]]
            
        else:
            # Default to random selection
            selected = np.random.choice(active_clients, size=num_selected, replace=False).tolist()
            
        # Update client participation counts
        for client_id in selected:
            self.clients[client_id].round_participation += 1
            
        return selected
        
    def _is_client_eligible(self, client_info: ClientInfo) -> bool:
        """Check if client is eligible for training"""
        # Check if client has sufficient privacy budget
        if self.config.enable_privacy and client_info.privacy_budget_remaining <= 0:
            return False
            
        # Check if client is not marked as Byzantine
        if client_info.byzantine_score > self.config.byzantine_threshold:
            return False
            
        # Check last seen time (client should be active)
        time_since_last_seen = time.time() - client_info.last_seen
        if time_since_last_seen > 3600:  # 1 hour timeout
            client_info.is_active = False
            return False
            
        return True
        
    def _send_training_request(
        self,
        client_id: str,
        global_weights: Dict[str, torch.Tensor],
        round_id: int
    ) -> Optional[Dict[str, Any]]:
        """Send training request to a specific client"""
        try:
            # Prepare training request
            request = {
                "message_type": MessageType.TRAINING_REQUEST,
                "round_id": round_id,
                "global_weights": global_weights,
                "training_config": {
                    "epochs": self.config.client_epochs,
                    "batch_size": self.config.client_batch_size,
                    "learning_rate": self.config.learning_rate
                },
                "privacy_budget": self.clients[client_id].privacy_budget_remaining if self.config.enable_privacy else None
            }
            
            # Send request via secure communicator
            response = self.communicator.send_message(client_id, request, timeout=self.config.communication_rounds_timeout)
            
            if response and response.get("success"):
                # Update client last seen
                with self.client_lock:
                    self.clients[client_id].last_seen = time.time()
                    
                return response.get("client_update")
            else:
                logger.warning(f"Failed to get response from client {client_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending training request to client {client_id}: {str(e)}")
            return None
            
    def _aggregate_client_updates(
        self,
        client_updates: Dict[str, Any],
        client_data_sizes: List[int]
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Aggregate client updates using the configured strategy"""
        
        # Extract weights from client updates
        client_weights = []
        for client_id, update in client_updates.items():
            weights = update.get("weights", {})
            client_weights.append(weights)
            
        # Perform aggregation
        aggregated_weights, privacy_spent = self.aggregator.aggregate(
            client_weights=client_weights,
            client_data_sizes=client_data_sizes,
            round_id=self.current_round
        )
        
        return aggregated_weights, privacy_spent
        
    def _filter_byzantine_updates(self, client_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out potentially Byzantine (malicious) client updates"""
        if len(client_updates) < 3:
            # Need at least 3 clients for Byzantine detection
            return client_updates
            
        # Extract weight updates for analysis
        weight_updates = []
        client_ids = list(client_updates.keys())
        
        for client_id in client_ids:
            update = client_updates[client_id]
            weights = update.get("weights", {})
            
            # Flatten weights for analysis
            flattened_weights = []
            for param_name, param_tensor in weights.items():
                flattened_weights.extend(param_tensor.flatten().tolist())
                
            weight_updates.append(flattened_weights)
            
        # Convert to numpy array for analysis
        weight_updates = np.array(weight_updates)
        
        # Detect outliers using statistical methods
        mean_update = np.mean(weight_updates, axis=0)
        distances = []
        
        for i, update in enumerate(weight_updates):
            distance = np.linalg.norm(update - mean_update)
            distances.append((client_ids[i], distance))
            
        # Sort by distance and identify outliers
        distances.sort(key=lambda x: x[1])
        median_distance = np.median([d[1] for d in distances])
        
        filtered_updates = {}
        byzantine_clients = []
        
        for client_id, distance in distances:
            # Use median absolute deviation for outlier detection
            if distance <= median_distance * self.config.anomaly_detection_threshold:
                filtered_updates[client_id] = client_updates[client_id]
            else:
                byzantine_clients.append(client_id)
                # Increase Byzantine score
                with self.client_lock:
                    self.clients[client_id].byzantine_score += 0.1
                    
        if byzantine_clients:
            logger.warning(f"Filtered out potential Byzantine clients: {byzantine_clients}")
            
        return filtered_updates
        
    def _get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights"""
        with self.model_lock:
            weights = {}
            for name, param in self.global_model.named_parameters():
                weights[name] = param.data.clone()
            return weights
            
    def _update_global_model(self, aggregated_weights: Dict[str, torch.Tensor]):
        """Update global model with aggregated weights"""
        for name, param in self.global_model.named_parameters():
            if name in aggregated_weights:
                param.data = aggregated_weights[name]
                
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model performance"""
        # This would typically evaluate on a held-out validation set
        # For demonstration, return mock metrics
        return {
            "loss": np.random.uniform(0.3, 0.8),
            "accuracy": np.random.uniform(0.7, 0.9)
        }
        
    def _handle_client_failure(self, client_id: str):
        """Handle client failure during training"""
        with self.client_lock:
            if client_id in self.clients:
                client_info = self.clients[client_id]
                client_info.is_active = False
                client_info.byzantine_score += 0.05  # Small penalty for failure
                logger.warning(f"Client {client_id} marked as inactive due to failure")
                
    def _update_client_histories(self, client_updates: Dict[str, Any]):
        """Update client performance histories"""
        with self.client_lock:
            for client_id, update in client_updates.items():
                if client_id in self.clients:
                    client_info = self.clients[client_id]
                    
                    # Extract performance metrics
                    accuracy = update.get("metrics", {}).get("accuracy", 0.0)
                    client_info.performance_history.append(accuracy)
                    
                    # Keep only recent history
                    if len(client_info.performance_history) > 10:
                        client_info.performance_history = client_info.performance_history[-10:]
                        
                    # Update privacy budget if applicable
                    if self.config.enable_privacy:
                        privacy_used = update.get("privacy_used", 0.0)
                        client_info.privacy_budget_remaining -= privacy_used
                        
    def _update_performance_metrics(self, fed_round: FederatedRound):
        """Update performance metrics for monitoring"""
        if fed_round.global_accuracy is not None:
            self.performance_metrics["global_accuracy"].append(fed_round.global_accuracy)
            
        if fed_round.global_loss is not None:
            self.performance_metrics["global_loss"].append(fed_round.global_loss)
            
        self.performance_metrics["num_participants"].append(len(fed_round.selected_clients))
        self.performance_metrics["privacy_spent"].append(fed_round.privacy_spent)
        
        round_duration = fed_round.end_time - fed_round.start_time if fed_round.end_time else 0
        self.performance_metrics["round_duration"].append(round_duration)
        
    def _save_checkpoint(self, round_id: int):
        """Save training checkpoint"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "round_id": round_id,
            "global_model_state": self.global_model.state_dict(),
            "clients": {cid: asdict(cinfo) for cid, cinfo in self.clients.items()},
            "performance_metrics": dict(self.performance_metrics),
            "training_history": [asdict(r) for r in self.training_history]
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_round_{round_id}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def _save_final_results(self):
        """Save final training results and generate report"""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save training history
        history_path = results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            history_data = []
            for round_info in self.training_history:
                round_dict = asdict(round_info)
                # Convert tensors to lists for JSON serialization
                if round_dict.get("aggregated_weights"):
                    round_dict["aggregated_weights"] = "saved_separately"
                history_data.append(round_dict)
            json.dump(history_data, f, indent=2, default=str)
            
        # Save performance metrics
        metrics_path = results_dir / "performance_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.performance_metrics), f, indent=2, default=str)
            
        # Save client information
        clients_path = results_dir / "client_info.json"
        with open(clients_path, 'w') as f:
            client_data = {cid: asdict(cinfo) for cid, cinfo in self.clients.items()}
            json.dump(client_data, f, indent=2, default=str)
            
        # Save final model
        model_path = results_dir / "final_global_model.pt"
        torch.save(self.global_model.state_dict(), model_path)
        
        # Generate summary report
        self._generate_training_report()
        
        logger.info(f"Final results saved to {results_dir}")
        
    def _generate_training_report(self):
        """Generate comprehensive training report"""
        report_path = self.output_dir / "federated_training_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Federated Learning Training Report\n\n")
            f.write(f"**Server ID:** {self.server_id}\n")
            f.write(f"**Training Completed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training summary
            f.write("## Training Summary\n\n")
            f.write(f"- **Total Rounds:** {len(self.training_history)}\n")
            f.write(f"- **Total Clients:** {len(self.clients)}\n")
            f.write(f"- **Training Duration:** {time.time() - self.training_start_time:.2f} seconds\n")
            
            if self.performance_metrics.get("global_accuracy"):
                final_accuracy = self.performance_metrics["global_accuracy"][-1]
                f.write(f"- **Final Global Accuracy:** {final_accuracy:.4f}\n")
                
            if self.config.enable_privacy:
                total_privacy_spent = sum(self.performance_metrics.get("privacy_spent", []))
                f.write(f"- **Total Privacy Budget Spent:** {total_privacy_spent:.6f}\n")
                
            # Client statistics
            f.write("\n## Client Statistics\n\n")
            active_clients = sum(1 for c in self.clients.values() if c.is_active)
            f.write(f"- **Active Clients:** {active_clients}/{len(self.clients)}\n")
            
            avg_participation = np.mean([c.round_participation for c in self.clients.values()])
            f.write(f"- **Average Participation:** {avg_participation:.1f} rounds\n")
            
            # Performance trends
            if self.performance_metrics.get("global_accuracy"):
                f.write("\n## Performance Trends\n\n")
                accuracies = self.performance_metrics["global_accuracy"]
                f.write(f"- **Initial Accuracy:** {accuracies[0]:.4f}\n")
                f.write(f"- **Final Accuracy:** {accuracies[-1]:.4f}\n")
                f.write(f"- **Improvement:** {accuracies[-1] - accuracies[0]:.4f}\n")
                
        logger.info(f"Training report generated: {report_path}")
        
    async def _process_async_updates(self):
        """Process client updates in asynchronous mode"""
        while True:
            try:
                # Wait for client update
                client_update = await self.update_queue.get()
                client_id = client_update["client_id"]
                
                # Check staleness
                current_staleness = self.staleness_tracker.get(client_id, 0)
                if current_staleness > self.config.async_staleness_threshold:
                    logger.warning(f"Dropping stale update from client {client_id} (staleness: {current_staleness})")
                    continue
                    
                # Process update immediately
                weights = client_update["weights"]
                data_size = client_update.get("data_size", 1)
                
                # Apply update to global model with staleness compensation
                staleness_weight = 1.0 / (1.0 + current_staleness)
                
                with self.model_lock:
                    for name, param in self.global_model.named_parameters():
                        if name in weights:
                            update = weights[name] * staleness_weight
                            param.data += self.config.learning_rate * update
                            
                # Update staleness tracker
                self.staleness_tracker[client_id] = 0
                for other_client in self.staleness_tracker:
                    if other_client != client_id:
                        self.staleness_tracker[other_client] += 1
                        
                self.current_round += 1
                logger.debug(f"Applied async update from client {client_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing async update: {str(e)}")
                
    async def _broadcast_async_training_request(self):
        """Broadcast training request to all clients in async mode"""
        active_clients = [cid for cid, cinfo in self.clients.items() if cinfo.is_active]
        
        for client_id in active_clients:
            try:
                global_weights = self._get_global_weights()
                request = {
                    "message_type": MessageType.ASYNC_TRAINING_REQUEST,
                    "global_weights": global_weights,
                    "training_config": {
                        "epochs": self.config.client_epochs,
                        "batch_size": self.config.client_batch_size,
                        "learning_rate": self.config.learning_rate
                    }
                }
                
                # Send non-blocking request
                self.communicator.send_message_async(client_id, request)
                
            except Exception as e:
                logger.error(f"Error sending async training request to {client_id}: {str(e)}")
                
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status"""
        with self.client_lock:
            active_clients = sum(1 for c in self.clients.values() if c.is_active)
            
        return {
            "server_id": self.server_id,
            "is_training": self.is_training,
            "current_round": self.current_round,
            "total_clients": len(self.clients),
            "active_clients": active_clients,
            "training_mode": "async" if self.config.enable_async_mode else "sync",
            "privacy_enabled": self.config.enable_privacy,
            "byzantine_tolerance_enabled": self.config.enable_byzantine_tolerance
        }
        
    def stop_training(self):
        """Stop federated training gracefully"""
        if self.is_training:
            logger.info("Stopping federated training...")
            self.is_training = False
            self._save_final_results()
            
    def cleanup(self):
        """Clean up server resources"""
        self.stop_training()
        self.thread_pool.shutdown(wait=True)
        if hasattr(self, 'communicator'):
            self.communicator.cleanup()
        logger.info("Federated server cleanup completed")


# Example usage and testing

def create_example_federated_server():
    """Create an example federated server for testing"""
    
    config = FederatedServerConfig(
        server_id="example_server",
        max_clients=10,
        min_clients_per_round=2,
        max_clients_per_round=5,
        global_rounds=20,
        enable_privacy=True,
        enable_byzantine_tolerance=True
    )
    
    model_config = {
        "input_dim": 768,
        "output_dim": 768,
        "rank": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj"]
    }
    
    server = FederatedServer(
        config=config,
        model_config=model_config,
        output_dir="example_federated_server"
    )
    
    return server


if __name__ == "__main__":
    # Example server creation and basic testing
    server = create_example_federated_server()
    
    print("Federated server created successfully!")
    print(f"Server status: {server.get_server_status()}")
    
    # Clean up
    server.cleanup()