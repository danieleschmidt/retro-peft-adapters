"""
Data models for retro-peft adapters database.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class AdapterType(Enum):
    """Supported adapter types"""
    RETRO_LORA = "retro_lora"
    RETRO_ADALORA = "retro_adalora"
    RETRO_IA3 = "retro_ia3"
    RETRO_PREFIX = "retro_prefix"


class TrainingStatus(Enum):
    """Training run status"""
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(Enum):
    """Performance metric types"""
    TRAINING = "training"
    VALIDATION = "validation"
    INFERENCE = "inference"
    RETRIEVAL = "retrieval"
    SYSTEM = "system"


@dataclass
class AdapterConfig:
    """
    Configuration for a retro-peft adapter.
    """
    id: Optional[int] = None
    name: str = ""
    adapter_type: str = AdapterType.RETRO_LORA.value
    base_model: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.name:
            raise ValueError("Adapter name is required")
        
        if self.adapter_type not in [t.value for t in AdapterType]:
            raise ValueError(f"Invalid adapter type: {self.adapter_type}")
        
        if not self.base_model:
            raise ValueError("Base model is required")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdapterConfig':
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            name=data['name'],
            adapter_type=data['adapter_type'],
            base_model=data['base_model'],
            config=data.get('config', {}),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'adapter_type': self.adapter_type,
            'base_model': self.base_model,
            'config': self.config,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    def get_target_modules(self) -> List[str]:
        """Get target modules from config"""
        return self.config.get('target_modules', [])
    
    def get_rank(self) -> int:
        """Get adapter rank from config"""
        if self.adapter_type == AdapterType.RETRO_LORA.value:
            return self.config.get('r', 16)
        elif self.adapter_type == AdapterType.RETRO_ADALORA.value:
            return self.config.get('target_r', 8)
        else:
            return 0
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval-specific configuration"""
        return {
            'retrieval_dim': self.config.get('retrieval_dim', 768),
            'retrieval_layers': self.config.get('retrieval_layers', []),
            'fusion_method': self.config.get('fusion_method', 'cross_attention'),
            'max_retrieved_docs': self.config.get('max_retrieved_docs', 5),
            'retrieval_weight': self.config.get('retrieval_weight', 0.3)
        }


@dataclass
class TrainingRun:
    """
    Training run metadata and results.
    """
    id: Optional[int] = None
    adapter_config_id: int = 0
    run_name: str = ""
    status: str = TrainingStatus.STARTED.value
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate training run after initialization"""
        if not self.run_name:
            raise ValueError("Run name is required")
        
        if self.adapter_config_id <= 0:
            raise ValueError("Valid adapter config ID is required")
        
        if self.status not in [s.value for s in TrainingStatus]:
            raise ValueError(f"Invalid training status: {self.status}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingRun':
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            adapter_config_id=data['adapter_config_id'],
            run_name=data['run_name'],
            status=data.get('status', TrainingStatus.STARTED.value),
            hyperparameters=data.get('hyperparameters', {}),
            metrics=data.get('metrics', {}),
            checkpoint_path=data.get('checkpoint_path'),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'adapter_config_id': self.adapter_config_id,
            'run_name': self.run_name,
            'status': self.status,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'checkpoint_path': self.checkpoint_path,
            'started_at': self.started_at,
            'completed_at': self.completed_at
        }
    
    def is_completed(self) -> bool:
        """Check if training run is completed"""
        return self.status == TrainingStatus.COMPLETED.value
    
    def is_running(self) -> bool:
        """Check if training run is currently running"""
        return self.status in [TrainingStatus.STARTED.value, TrainingStatus.RUNNING.value]
    
    def get_duration_seconds(self) -> Optional[float]:
        """Get training duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def get_final_metrics(self) -> Dict[str, Any]:
        """Get final training metrics"""
        return self.metrics.get('final', {})
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics achieved during training"""
        return self.metrics.get('best', {})


@dataclass
class RetrievalIndex:
    """
    Retrieval index metadata and configuration.
    """
    id: Optional[int] = None
    name: str = ""
    backend_type: str = "faiss"
    index_path: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    num_documents: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate retrieval index after initialization"""
        if not self.name:
            raise ValueError("Index name is required")
        
        if not self.index_path:
            raise ValueError("Index path is required")
        
        if not self.embedding_model:
            raise ValueError("Embedding model is required")
        
        if self.embedding_dim <= 0:
            raise ValueError("Valid embedding dimension is required")
        
        if self.num_documents < 0:
            raise ValueError("Number of documents cannot be negative")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalIndex':
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            name=data['name'],
            backend_type=data.get('backend_type', 'faiss'),
            index_path=data['index_path'],
            embedding_model=data['embedding_model'],
            embedding_dim=data['embedding_dim'],
            num_documents=data['num_documents'],
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at'),
            last_updated=data.get('last_updated')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'backend_type': self.backend_type,
            'index_path': self.index_path,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'num_documents': self.num_documents,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }
    
    def get_chunk_size(self) -> int:
        """Get chunk size from metadata"""
        return self.metadata.get('chunk_size', 512)
    
    def get_overlap(self) -> int:
        """Get chunk overlap from metadata"""
        return self.metadata.get('overlap', 50)
    
    def get_index_size_mb(self) -> Optional[float]:
        """Get index size in MB"""
        size_bytes = self.metadata.get('index_size_bytes')
        if size_bytes:
            return size_bytes / (1024 * 1024)
        return None
    
    def supports_filtering(self) -> bool:
        """Check if backend supports metadata filtering"""
        return self.backend_type in ['qdrant', 'weaviate']


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for training and inference.
    """
    id: Optional[int] = None
    training_run_id: Optional[int] = None
    metric_type: str = MetricType.TRAINING.value
    metric_name: str = ""
    metric_value: float = 0.0
    step: Optional[int] = None
    epoch: Optional[int] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate performance metrics after initialization"""
        if not self.metric_name:
            raise ValueError("Metric name is required")
        
        if self.metric_type not in [t.value for t in MetricType]:
            raise ValueError(f"Invalid metric type: {self.metric_type}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            training_run_id=data.get('training_run_id'),
            metric_type=data.get('metric_type', MetricType.TRAINING.value),
            metric_name=data['metric_name'],
            metric_value=data['metric_value'],
            step=data.get('step'),
            epoch=data.get('epoch'),
            timestamp=data.get('timestamp'),
            metadata=data.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'training_run_id': self.training_run_id,
            'metric_type': self.metric_type,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'step': self.step,
            'epoch': self.epoch,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def is_training_metric(self) -> bool:
        """Check if this is a training metric"""
        return self.metric_type == MetricType.TRAINING.value
    
    def is_validation_metric(self) -> bool:
        """Check if this is a validation metric"""
        return self.metric_type == MetricType.VALIDATION.value
    
    def is_retrieval_metric(self) -> bool:
        """Check if this is a retrieval metric"""
        return self.metric_type == MetricType.RETRIEVAL.value
    
    def get_formatted_value(self, precision: int = 4) -> str:
        """Get formatted metric value"""
        if isinstance(self.metric_value, float):
            return f"{self.metric_value:.{precision}f}"
        return str(self.metric_value)


@dataclass
class ModelVersion:
    """
    Model version with performance summary and tags.
    """
    id: Optional[int] = None
    adapter_config_id: int = 0
    version: str = ""
    checkpoint_path: str = ""
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate model version after initialization"""
        if not self.version:
            raise ValueError("Version is required")
        
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path is required")
        
        if self.adapter_config_id <= 0:
            raise ValueError("Valid adapter config ID is required")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            adapter_config_id=data['adapter_config_id'],
            version=data['version'],
            checkpoint_path=data['checkpoint_path'],
            performance_summary=data.get('performance_summary', {}),
            tags=data.get('tags', []),
            created_at=data.get('created_at')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'adapter_config_id': self.adapter_config_id,
            'version': self.version,
            'checkpoint_path': self.checkpoint_path,
            'performance_summary': self.performance_summary,
            'tags': self.tags,
            'created_at': self.created_at
        }
    
    def add_tag(self, tag: str):
        """Add tag to model version"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str):
        """Remove tag from model version"""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if model version has tag"""
        return tag in self.tags
    
    def get_accuracy(self) -> Optional[float]:
        """Get accuracy from performance summary"""
        return self.performance_summary.get('accuracy')
    
    def get_loss(self) -> Optional[float]:
        """Get loss from performance summary"""
        return self.performance_summary.get('loss')
    
    def get_retrieval_metrics(self) -> Dict[str, float]:
        """Get retrieval metrics from performance summary"""
        return self.performance_summary.get('retrieval', {})