"""
Repository classes for data access operations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .connection import DatabaseManager
from .models import AdapterConfig, ModelVersion, PerformanceMetrics, RetrievalIndex, TrainingRun


class BaseRepository:
    """Base repository with common database operations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager


class AdapterRepository(BaseRepository):
    """Repository for adapter configurations"""

    def create(self, adapter: AdapterConfig) -> int:
        """Create new adapter configuration"""
        adapter_id = self.db.create_adapter_config(
            name=adapter.name,
            adapter_type=adapter.adapter_type,
            base_model=adapter.base_model,
            config=adapter.config,
        )
        adapter.id = adapter_id
        return adapter_id

    def get_by_name(self, name: str) -> Optional[AdapterConfig]:
        """Get adapter by name"""
        data = self.db.get_adapter_config(name)
        if data:
            return AdapterConfig.from_dict(data)
        return None

    def get_by_id(self, adapter_id: int) -> Optional[AdapterConfig]:
        """Get adapter by ID"""
        query = "SELECT * FROM adapter_configs WHERE id = ?"
        data = self.db.execute_query(query, (adapter_id,), fetch="one")
        if data:
            return AdapterConfig.from_dict(data)
        return None

    def list_all(self) -> List[AdapterConfig]:
        """List all adapter configurations"""
        query = "SELECT * FROM adapter_configs ORDER BY created_at DESC"
        results = self.db.execute_query(query, fetch="all")
        return [AdapterConfig.from_dict(data) for data in results]

    def update_config(self, adapter_id: int, config: Dict[str, Any]):
        """Update adapter configuration"""
        query = """
            UPDATE adapter_configs 
            SET config_json = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """
        import json

        self.db.execute_query(query, (json.dumps(config), adapter_id))

    def delete(self, adapter_id: int):
        """Delete adapter configuration"""
        query = "DELETE FROM adapter_configs WHERE id = ?"
        self.db.execute_query(query, (adapter_id,))

    def search_by_model(self, base_model: str) -> List[AdapterConfig]:
        """Search adapters by base model"""
        query = "SELECT * FROM adapter_configs WHERE base_model = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (base_model,), fetch="all")
        return [AdapterConfig.from_dict(data) for data in results]


class TrainingRepository(BaseRepository):
    """Repository for training runs"""

    def create(self, training_run: TrainingRun) -> int:
        """Create new training run"""
        run_id = self.db.create_training_run(
            adapter_config_id=training_run.adapter_config_id,
            run_name=training_run.run_name,
            hyperparameters=training_run.hyperparameters,
        )
        training_run.id = run_id
        return run_id

    def get_by_id(self, run_id: int) -> Optional[TrainingRun]:
        """Get training run by ID"""
        query = "SELECT * FROM training_runs WHERE id = ?"
        data = self.db.execute_query(query, (run_id,), fetch="one")
        if data:
            import json

            data["hyperparameters"] = json.loads(data["hyperparameters_json"])
            data["metrics"] = json.loads(data["metrics_json"])
            return TrainingRun.from_dict(data)
        return None

    def update_status(
        self,
        run_id: int,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Update training run status"""
        self.db.update_training_run_status(run_id, status, metrics, checkpoint_path)

    def get_history(self, adapter_config_id: int) -> List[TrainingRun]:
        """Get training history for adapter"""
        results = self.db.get_training_history(adapter_config_id)
        return [TrainingRun.from_dict(data) for data in results]

    def get_active_runs(self) -> List[TrainingRun]:
        """Get currently active training runs"""
        query = """
            SELECT * FROM training_runs 
            WHERE status IN ('started', 'running') 
            ORDER BY started_at DESC
        """
        results = self.db.execute_query(query, fetch="all")
        training_runs = []
        for data in results:
            import json

            data["hyperparameters"] = json.loads(data["hyperparameters_json"])
            data["metrics"] = json.loads(data["metrics_json"])
            training_runs.append(TrainingRun.from_dict(data))
        return training_runs

    def get_completed_runs(
        self, adapter_config_id: Optional[int] = None, limit: int = 10
    ) -> List[TrainingRun]:
        """Get completed training runs"""
        if adapter_config_id:
            query = """
                SELECT * FROM training_runs 
                WHERE adapter_config_id = ? AND status = 'completed'
                ORDER BY completed_at DESC LIMIT ?
            """
            params = (adapter_config_id, limit)
        else:
            query = """
                SELECT * FROM training_runs 
                WHERE status = 'completed'
                ORDER BY completed_at DESC LIMIT ?
            """
            params = (limit,)

        results = self.db.execute_query(query, params, fetch="all")
        training_runs = []
        for data in results:
            import json

            data["hyperparameters"] = json.loads(data["hyperparameters_json"])
            data["metrics"] = json.loads(data["metrics_json"])
            training_runs.append(TrainingRun.from_dict(data))
        return training_runs


class RetrievalRepository(BaseRepository):
    """Repository for retrieval indices"""

    def create(self, index: RetrievalIndex) -> int:
        """Create new retrieval index"""
        index_id = self.db.create_retrieval_index(
            name=index.name,
            backend_type=index.backend_type,
            index_path=index.index_path,
            embedding_model=index.embedding_model,
            embedding_dim=index.embedding_dim,
            num_documents=index.num_documents,
            metadata=index.metadata,
        )
        index.id = index_id
        return index_id

    def get_by_name(self, name: str) -> Optional[RetrievalIndex]:
        """Get retrieval index by name"""
        data = self.db.get_retrieval_index(name)
        if data:
            return RetrievalIndex.from_dict(data)
        return None

    def get_by_id(self, index_id: int) -> Optional[RetrievalIndex]:
        """Get retrieval index by ID"""
        query = "SELECT * FROM retrieval_indices WHERE id = ?"
        data = self.db.execute_query(query, (index_id,), fetch="one")
        if data:
            import json

            data["metadata"] = json.loads(data["metadata_json"])
            return RetrievalIndex.from_dict(data)
        return None

    def list_by_backend(self, backend_type: str) -> List[RetrievalIndex]:
        """List indices by backend type"""
        query = "SELECT * FROM retrieval_indices WHERE backend_type = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (backend_type,), fetch="all")
        indices = []
        for data in results:
            import json

            data["metadata"] = json.loads(data["metadata_json"])
            indices.append(RetrievalIndex.from_dict(data))
        return indices

    def update_metadata(self, index_id: int, metadata: Dict[str, Any]):
        """Update index metadata"""
        query = """
            UPDATE retrieval_indices 
            SET metadata_json = ?, last_updated = CURRENT_TIMESTAMP 
            WHERE id = ?
        """
        import json

        self.db.execute_query(query, (json.dumps(metadata), index_id))

    def delete(self, index_id: int):
        """Delete retrieval index"""
        query = "DELETE FROM retrieval_indices WHERE id = ?"
        self.db.execute_query(query, (index_id,))


class MetricsRepository(BaseRepository):
    """Repository for performance metrics"""

    def log_metric(self, metric: PerformanceMetrics):
        """Log performance metric"""
        self.db.log_performance_metric(
            training_run_id=metric.training_run_id,
            metric_type=metric.metric_type,
            metric_name=metric.metric_name,
            metric_value=metric.metric_value,
            step=metric.step,
            epoch=metric.epoch,
            metadata=metric.metadata,
        )

    def get_metrics(
        self, training_run_id: int, metric_type: Optional[str] = None
    ) -> List[PerformanceMetrics]:
        """Get metrics for training run"""
        results = self.db.get_performance_metrics(training_run_id, metric_type)
        return [PerformanceMetrics.from_dict(data) for data in results]

    def get_metric_history(
        self, training_run_id: int, metric_name: str
    ) -> List[PerformanceMetrics]:
        """Get metric history over time"""
        query = """
            SELECT * FROM performance_metrics 
            WHERE training_run_id = ? AND metric_name = ?
            ORDER BY timestamp
        """
        results = self.db.execute_query(query, (training_run_id, metric_name), fetch="all")
        metrics = []
        for data in results:
            import json

            data["metadata"] = json.loads(data["metadata_json"])
            metrics.append(PerformanceMetrics.from_dict(data))
        return metrics

    def get_latest_metrics(
        self, training_run_id: int, metric_type: str = "validation"
    ) -> Dict[str, float]:
        """Get latest metrics for a training run"""
        query = """
            SELECT metric_name, metric_value 
            FROM performance_metrics 
            WHERE training_run_id = ? AND metric_type = ?
            ORDER BY timestamp DESC
        """
        results = self.db.execute_query(query, (training_run_id, metric_type), fetch="all")

        # Get most recent value for each metric
        latest_metrics = {}
        seen_metrics = set()

        for result in results:
            metric_name = result["metric_name"]
            if metric_name not in seen_metrics:
                latest_metrics[metric_name] = result["metric_value"]
                seen_metrics.add(metric_name)

        return latest_metrics

    def get_best_metrics(
        self, training_run_id: int, metric_name: str, higher_is_better: bool = True
    ) -> Optional[PerformanceMetrics]:
        """Get best metric value for training run"""
        order_by = "DESC" if higher_is_better else "ASC"
        query = f"""
            SELECT * FROM performance_metrics 
            WHERE training_run_id = ? AND metric_name = ?
            ORDER BY metric_value {order_by} LIMIT 1
        """
        data = self.db.execute_query(query, (training_run_id, metric_name), fetch="one")
        if data:
            import json

            data["metadata"] = json.loads(data["metadata_json"])
            return PerformanceMetrics.from_dict(data)
        return None

    def cleanup_old_metrics(self, days: int = 30):
        """Clean up old metrics"""
        self.db.cleanup_old_metrics(days)
