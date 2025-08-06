"""
Federated Learning Client for Retro-PEFT-Adapters

Implements the client-side logic for federated learning with advanced features:
- Privacy-preserving local training with differential privacy
- Secure communication with the federated server
- Local data management and validation
- Performance monitoring and adaptive training
- Resource management and optimization
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from .privacy import PrivacyEngine, PrivacyConfig
from .communication import SecureCommunicator, MessageType

logger = logging.getLogger(__name__)


@dataclass
class FederatedClientConfig:
    """Configuration for federated learning client"""
    client_id: str
    server_address: str = "localhost"
    server_port: int = 8080
    
    # Training parameters
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer_type: str = "adam"  # adam, sgd, adamw
    
    # Privacy parameters
    enable_privacy: bool = True
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Communication parameters
    heartbeat_interval: float = 60.0  # seconds
    communication_timeout: float = 300.0  # 5 minutes
    max_retries: int = 3
    compression_enabled: bool = True
    
    # Resource management
    max_memory_usage_gb: float = 4.0
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.5
    
    # Data management
    data_validation_enabled: bool = True
    cache_processed_data: bool = True
    
    # Monitoring
    enable_performance_monitoring: bool = True
    save_local_metrics: bool = True


@dataclass
class TrainingMetrics:
    """Metrics collected during local training"""
    round_id: int
    local_loss: float
    local_accuracy: float
    training_time: float
    data_size: int
    privacy_used: float
    memory_peak_mb: float
    gpu_utilization: float


class FederatedDataset(Dataset):
    """Custom dataset wrapper for federated learning"""
    
    def __init__(self, data: List[Any], labels: List[Any], transform: Optional[Callable] = None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label


class FederatedClient:
    """
    Advanced federated learning client for retro-peft adapters
    
    Features:
    - Privacy-preserving local training
    - Secure communication with server
    - Adaptive resource management
    - Performance monitoring and optimization
    - Local model validation
    - Automatic recovery and resilience
    """
    
    def __init__(
        self,
        config: FederatedClientConfig,
        local_dataset: Dataset,
        model_factory: Callable = None,
        output_dir: str = None
    ):
        self.config = config
        self.local_dataset = local_dataset
        self.model_factory = model_factory
        
        # Setup output directory
        if output_dir is None:
            output_dir = f"federated_client_{config.client_id}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Client state
        self.client_id = config.client_id
        self.is_active = False
        self.is_training = False
        self.current_round = 0
        self.registration_time: Optional[float] = None
        
        # Model and training components
        self.local_model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.device = self._setup_device()
        
        # Privacy engine
        if config.enable_privacy:
            privacy_config = PrivacyConfig(
                epsilon=config.privacy_epsilon,
                delta=config.privacy_delta,
                max_grad_norm=config.max_grad_norm
            )
            self.privacy_engine = PrivacyEngine(
                epsilon=config.privacy_epsilon,
                delta=config.privacy_delta
            )
        else:
            self.privacy_engine = None
            
        # Communication
        self.communicator = SecureCommunicator(
            node_id=self.client_id,
            use_encryption=True
        )
        
        # Training history
        self.training_metrics: List[TrainingMetrics] = []
        self.performance_history: Dict[str, List[float]] = {
            "local_loss": [],
            "local_accuracy": [],
            "training_time": [],
            "memory_usage": [],
            "privacy_used": []
        }
        
        # Threading
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.training_lock = threading.Lock()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Federated client initialized: {self.client_id}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dataset size: {len(local_dataset)}")
        
    def _setup_device(self) -> torch.device:
        """Setup compute device (CPU/GPU)"""
        if self.config.enable_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            # Set memory fraction if specified
            if self.config.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for training")
            
        return device
        
    def _setup_logging(self):
        """Setup detailed logging for client"""
        log_file = self.output_dir / f"client_{self.client_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        
    def connect_to_server(self) -> bool:
        """
        Connect and register with the federated server
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Connecting to federated server at {self.config.server_address}:{self.config.server_port}")
        
        try:
            # Prepare client information
            client_info = {
                "client_id": self.client_id,
                "data_size": len(self.local_dataset),
                "compute_capacity": self._estimate_compute_capacity(),
                "network_bandwidth": 1.0,  # Placeholder
                "privacy_enabled": self.config.enable_privacy,
                "device_type": str(self.device)
            }
            
            # Send registration request
            registration_request = {
                "message_type": MessageType.CLIENT_REGISTRATION,
                "client_info": client_info
            }
            
            server_id = f"server_{self.config.server_address}_{self.config.server_port}"
            response = self.communicator.send_message(
                server_id, 
                registration_request,
                timeout=self.config.communication_timeout
            )
            
            if response and response.get("success"):
                self.is_active = True
                self.registration_time = time.time()
                
                # Extract server configuration
                server_config = response.get("training_config", {})
                self._update_training_config(server_config)
                
                # Start heartbeat thread
                self._start_heartbeat()
                
                logger.info(f"Successfully registered with server")
                logger.info(f"Server training config: {server_config}")
                
                return True
            else:
                error_msg = response.get("error", "Unknown error") if response else "No response from server"
                logger.error(f"Registration failed: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to server: {str(e)}")
            return False
            
    def _estimate_compute_capacity(self) -> float:
        """Estimate client compute capacity"""
        if self.device.type == "cuda":
            # GPU-based estimate
            gpu_properties = torch.cuda.get_device_properties(self.device)
            compute_capacity = gpu_properties.total_memory / (1024**3)  # GB as proxy
        else:
            # CPU-based estimate
            import psutil
            compute_capacity = psutil.cpu_count() * psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else 1.0
            
        return min(compute_capacity, 10.0)  # Cap at 10.0
        
    def _update_training_config(self, server_config: Dict[str, Any]):
        """Update local training configuration based on server requirements"""
        if "epochs" in server_config:
            self.config.local_epochs = server_config["epochs"]
        if "batch_size" in server_config:
            self.config.batch_size = server_config["batch_size"]
        if "learning_rate" in server_config:
            self.config.learning_rate = server_config["learning_rate"]
            
        logger.debug(f"Updated training config: epochs={self.config.local_epochs}, "
                    f"batch_size={self.config.batch_size}, lr={self.config.learning_rate}")
                    
    def _start_heartbeat(self):
        """Start heartbeat thread to maintain connection with server"""
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            logger.debug("Heartbeat thread started")
            
    def _heartbeat_loop(self):
        """Heartbeat loop to send periodic status updates to server"""
        while self.is_active:
            try:
                # Send heartbeat message
                heartbeat_msg = {
                    "message_type": MessageType.HEARTBEAT,
                    "client_id": self.client_id,
                    "timestamp": time.time(),
                    "status": "active",
                    "is_training": self.is_training,
                    "current_round": self.current_round
                }
                
                server_id = f"server_{self.config.server_address}_{self.config.server_port}"
                response = self.communicator.send_message(server_id, heartbeat_msg, timeout=30.0)
                
                if response and response.get("success"):
                    logger.debug("Heartbeat sent successfully")
                else:
                    logger.warning("Heartbeat failed - server may be unreachable")
                    
            except Exception as e:
                logger.error(f"Error in heartbeat: {str(e)}")
                
            # Wait for next heartbeat interval
            time.sleep(self.config.heartbeat_interval)
            
    def start_local_training(self, global_weights: Dict[str, torch.Tensor], round_id: int) -> Dict[str, Any]:
        """
        Perform local training with received global weights
        
        Args:
            global_weights: Global model weights from server
            round_id: Current federated learning round
            
        Returns:
            Dictionary containing local update and metrics
        """
        logger.info(f"Starting local training for round {round_id}")
        
        with self.training_lock:
            if self.is_training:
                logger.warning("Local training already in progress")
                return {"success": False, "error": "Training already in progress"}
                
            self.is_training = True
            self.current_round = round_id
            
        try:
            training_start_time = time.time()
            
            # Initialize/update local model
            if self.local_model is None:
                self._initialize_local_model()
                
            # Load global weights
            self._load_global_weights(global_weights)
            
            # Prepare data loader
            data_loader = self._create_data_loader()
            
            # Track resource usage
            initial_memory = self._get_memory_usage()
            
            # Local training loop
            local_metrics = self._train_local_model(data_loader, round_id)
            
            # Calculate resource usage
            final_memory = self._get_memory_usage()
            peak_memory = max(initial_memory, final_memory)
            
            # Get model weights after training
            local_weights = self._get_local_weights()
            
            # Calculate weight difference (update)
            weight_updates = {}
            for name in local_weights:
                if name in global_weights:
                    weight_updates[name] = local_weights[name] - global_weights[name]
                else:
                    weight_updates[name] = local_weights[name]
                    
            # Apply privacy if enabled
            privacy_used = 0.0
            if self.privacy_engine:
                weight_updates, privacy_used = self._apply_privacy_protection(weight_updates)
                
            training_end_time = time.time()
            training_duration = training_end_time - training_start_time
            
            # Create training metrics
            metrics = TrainingMetrics(
                round_id=round_id,
                local_loss=local_metrics["loss"],
                local_accuracy=local_metrics["accuracy"],
                training_time=training_duration,
                data_size=len(self.local_dataset),
                privacy_used=privacy_used,
                memory_peak_mb=peak_memory,
                gpu_utilization=local_metrics.get("gpu_utilization", 0.0)
            )
            
            self.training_metrics.append(metrics)
            self._update_performance_history(metrics)
            
            # Prepare response
            client_update = {
                "success": True,
                "client_id": self.client_id,
                "round_id": round_id,
                "weights": weight_updates,
                "data_size": len(self.local_dataset),
                "metrics": {
                    "loss": local_metrics["loss"],
                    "accuracy": local_metrics["accuracy"],
                    "training_time": training_duration
                },
                "privacy_used": privacy_used,
                "resource_usage": {
                    "memory_peak_mb": peak_memory,
                    "gpu_utilization": local_metrics.get("gpu_utilization", 0.0)
                }
            }
            
            logger.info(f"Local training completed - Loss: {local_metrics['loss']:.4f}, "
                       f"Accuracy: {local_metrics['accuracy']:.4f}, Time: {training_duration:.2f}s")
            
            # Save local metrics if enabled
            if self.config.save_local_metrics:
                self._save_local_metrics(metrics)
                
            return client_update
            
        except Exception as e:
            logger.error(f"Error during local training: {str(e)}")
            return {
                "success": False,
                "client_id": self.client_id,
                "round_id": round_id,
                "error": str(e)
            }
        finally:
            self.is_training = False
            
    def _initialize_local_model(self):
        """Initialize local model"""
        if self.model_factory:
            self.local_model = self.model_factory()
        else:
            # Default model initialization
            from ..adapters import RetroLoRA
            self.local_model = RetroLoRA(
                base_model=None,
                r=16,
                alpha=32,
                target_modules=["q_proj", "v_proj"]
            )
            
        self.local_model.to(self.device)
        
        # Initialize optimizer
        if self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.local_model.parameters(),
                lr=self.config.learning_rate
            )
        elif self.config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.local_model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            self.optimizer = optim.SGD(
                self.local_model.parameters(),
                lr=self.config.learning_rate
            )
            
        logger.debug("Local model and optimizer initialized")
        
    def _load_global_weights(self, global_weights: Dict[str, torch.Tensor]):
        """Load global weights into local model"""
        model_dict = self.local_model.state_dict()
        
        # Filter and load compatible weights
        compatible_weights = {}
        for name, weight in global_weights.items():
            if name in model_dict and model_dict[name].shape == weight.shape:
                compatible_weights[name] = weight.to(self.device)
            else:
                logger.warning(f"Skipping incompatible weight: {name}")
                
        model_dict.update(compatible_weights)
        self.local_model.load_state_dict(model_dict)
        
        logger.debug(f"Loaded {len(compatible_weights)} global weights")
        
    def _create_data_loader(self) -> DataLoader:
        """Create data loader for local training"""
        return DataLoader(
            self.local_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda"
        )
        
    def _train_local_model(self, data_loader: DataLoader, round_id: int) -> Dict[str, float]:
        """
        Train local model for specified number of epochs
        
        Args:
            data_loader: DataLoader for training data
            round_id: Current training round
            
        Returns:
            Training metrics dictionary
        """
        self.local_model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        gpu_utilization = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.local_model(data)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.local_model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                pred = outputs.argmax(dim=1)
                epoch_correct += pred.eq(targets).sum().item()
                epoch_samples += targets.size(0)
                
                # Monitor GPU utilization (if available)
                if self.device.type == "cuda":
                    gpu_utilization += torch.cuda.utilization() / 100.0
                    
            # Log epoch results
            epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
            avg_epoch_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else 0.0
            
            logger.debug(f"Epoch {epoch + 1}/{self.config.local_epochs} - "
                        f"Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            
        # Calculate final metrics
        avg_loss = total_loss / (len(data_loader) * self.config.local_epochs)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_gpu_utilization = gpu_utilization / (len(data_loader) * self.config.local_epochs) if self.device.type == "cuda" else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "gpu_utilization": avg_gpu_utilization
        }
        
    def _get_local_weights(self) -> Dict[str, torch.Tensor]:
        """Get current local model weights"""
        weights = {}
        for name, param in self.local_model.named_parameters():
            weights[name] = param.data.clone().cpu()
        return weights
        
    def _apply_privacy_protection(self, weight_updates: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float]:
        """Apply differential privacy to weight updates"""
        if not self.privacy_engine:
            return weight_updates, 0.0
            
        # Apply privacy mechanism to weight updates
        private_updates, privacy_used = self.privacy_engine.add_privacy_noise(
            weight_updates,
            sensitivity=1.0,  # L2 sensitivity
            round_id=self.current_round
        )
        
        return private_updates, privacy_used
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        else:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
            
    def _update_performance_history(self, metrics: TrainingMetrics):
        """Update performance history with new metrics"""
        self.performance_history["local_loss"].append(metrics.local_loss)
        self.performance_history["local_accuracy"].append(metrics.local_accuracy)
        self.performance_history["training_time"].append(metrics.training_time)
        self.performance_history["memory_usage"].append(metrics.memory_peak_mb)
        self.performance_history["privacy_used"].append(metrics.privacy_used)
        
        # Keep only recent history (last 50 rounds)
        for key in self.performance_history:
            if len(self.performance_history[key]) > 50:
                self.performance_history[key] = self.performance_history[key][-50:]
                
    def _save_local_metrics(self, metrics: TrainingMetrics):
        """Save local training metrics to disk"""
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        metrics_file = metrics_dir / f"round_{metrics.round_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, default=str)
            
    def evaluate_local_model(self, test_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate local model performance
        
        Args:
            test_dataset: Optional test dataset (uses training set if None)
            
        Returns:
            Evaluation metrics
        """
        if self.local_model is None:
            logger.warning("No local model available for evaluation")
            return {}
            
        eval_dataset = test_dataset if test_dataset is not None else self.local_dataset
        eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.local_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in eval_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.local_model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                total_correct += pred.eq(targets).sum().item()
                total_samples += targets.size(0)
                
        avg_loss = total_loss / len(eval_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples": total_samples
        }
        
    def get_client_status(self) -> Dict[str, Any]:
        """Get current client status"""
        return {
            "client_id": self.client_id,
            "is_active": self.is_active,
            "is_training": self.is_training,
            "current_round": self.current_round,
            "registration_time": self.registration_time,
            "dataset_size": len(self.local_dataset),
            "device": str(self.device),
            "privacy_enabled": self.config.enable_privacy,
            "total_training_rounds": len(self.training_metrics),
            "performance_summary": {
                "avg_accuracy": np.mean(self.performance_history["local_accuracy"]) if self.performance_history["local_accuracy"] else 0.0,
                "avg_loss": np.mean(self.performance_history["local_loss"]) if self.performance_history["local_loss"] else 0.0,
                "avg_training_time": np.mean(self.performance_history["training_time"]) if self.performance_history["training_time"] else 0.0
            }
        }
        
    def disconnect_from_server(self):
        """Disconnect from federated server"""
        logger.info("Disconnecting from federated server")
        
        self.is_active = False
        
        # Stop heartbeat thread
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5.0)
            
        # Send disconnection message
        try:
            disconnect_msg = {
                "message_type": MessageType.CLIENT_DISCONNECT,
                "client_id": self.client_id,
                "timestamp": time.time()
            }
            
            server_id = f"server_{self.config.server_address}_{self.config.server_port}"
            self.communicator.send_message(server_id, disconnect_msg, timeout=10.0)
            
        except Exception as e:
            logger.warning(f"Error sending disconnect message: {str(e)}")
            
        # Save final results
        self._save_final_results()
        
        logger.info("Client disconnected successfully")
        
    def _save_final_results(self):
        """Save final client results"""
        results_dir = self.output_dir / "final_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save training history
        history_file = results_dir / "training_history.json"
        with open(history_file, 'w') as f:
            history_data = [asdict(m) for m in self.training_metrics]
            json.dump(history_data, f, indent=2, default=str)
            
        # Save performance history
        performance_file = results_dir / "performance_history.json"
        with open(performance_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)
            
        # Save client configuration
        config_file = results_dir / "client_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
            
        # Save local model if available
        if self.local_model is not None:
            model_file = results_dir / "final_local_model.pt"
            torch.save(self.local_model.state_dict(), model_file)
            
        logger.info(f"Final results saved to {results_dir}")
        
    def cleanup(self):
        """Clean up client resources"""
        self.disconnect_from_server()
        
        if hasattr(self, 'communicator'):
            self.communicator.cleanup()
            
        # Clear GPU memory if used
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        logger.info("Client cleanup completed")


# Example usage and testing functions

def create_example_client(client_id: str = "example_client") -> FederatedClient:
    """Create an example federated client for testing"""
    
    # Create mock dataset
    mock_data = [torch.randn(128) for _ in range(1000)]
    mock_labels = [torch.randint(0, 10, (1,)).item() for _ in range(1000)]
    dataset = FederatedDataset(mock_data, mock_labels)
    
    # Client configuration
    config = FederatedClientConfig(
        client_id=client_id,
        server_address="localhost",
        server_port=8080,
        local_epochs=3,
        batch_size=32,
        learning_rate=1e-4,
        enable_privacy=True,
        privacy_epsilon=1.0
    )
    
    # Model factory
    def model_factory():
        from ..adapters import RetroLoRA
        return RetroLoRA(
            base_model=None,
            r=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"]
        )
        
    client = FederatedClient(
        config=config,
        local_dataset=dataset,
        model_factory=model_factory,
        output_dir=f"example_client_{client_id}"
    )
    
    return client


if __name__ == "__main__":
    # Example client creation and basic testing
    client = create_example_client("test_client_001")
    
    print("Federated client created successfully!")
    print(f"Client status: {client.get_client_status()}")
    
    # Clean up
    client.cleanup()