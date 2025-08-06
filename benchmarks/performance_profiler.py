"""
Performance Profiler for Retro-PEFT-Adapters

Advanced profiling system for detailed performance analysis including:
- Memory usage tracking with GPU memory optimization
- Latency profiling with percentile analysis  
- Throughput benchmarking under various loads
- Resource utilization monitoring
- Bottleneck identification and optimization recommendations
"""

import os
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import numpy as np
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ProfilerConfig:
    """Configuration for performance profiling"""
    sample_interval: float = 0.1  # seconds
    max_samples: int = 10000
    track_gpu_memory: bool = True
    track_cpu_usage: bool = True
    track_memory_usage: bool = True
    track_disk_io: bool = False
    track_network_io: bool = False
    export_detailed_logs: bool = True


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class LatencyMeasurement:
    """Latency measurement for a specific operation"""
    operation_name: str
    latency_ms: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThroughputMeasurement:
    """Throughput measurement over a time window"""
    window_start: float
    window_end: float
    operations_count: int
    throughput_ops_per_sec: float
    operation_type: str


class PerformanceProfiler:
    """
    Advanced performance profiler for retro-peft adapters
    
    Features:
    - Real-time resource monitoring
    - Latency percentile analysis  
    - Throughput benchmarking
    - Memory leak detection
    - GPU utilization tracking
    - Automated bottleneck identification
    - Performance regression detection
    """
    
    def __init__(self, config: ProfilerConfig = None):
        self.config = config or ProfilerConfig()
        
        # Data storage
        self.snapshots: deque = deque(maxlen=self.config.max_samples)
        self.latency_measurements: defaultdict = defaultdict(list)
        self.throughput_measurements: List[ThroughputMeasurement] = []
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Operation tracking
        self._operation_counts: defaultdict = defaultdict(int)
        self._operation_start_times: Dict[str, float] = {}
        
        # GPU availability
        self._gpu_available = self._check_gpu_availability()
        
        # Baseline measurements
        self._baseline_snapshot: Optional[PerformanceSnapshot] = None
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except ImportError:
            print("Warning: pynvml not available, GPU monitoring disabled")
            return False
        except Exception:
            return False
            
    def start_monitoring(self) -> None:
        """Start continuous performance monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        # Take baseline measurement
        self._baseline_snapshot = self._take_snapshot()
        
        print("Performance monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
        print("Performance monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread"""
        while self._monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.config.sample_interval)
            
    def _take_snapshot(self) -> PerformanceSnapshot:
        """Take a single performance snapshot"""
        current_time = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / (1024 * 1024)
        
        # GPU memory (if available)
        gpu_memory_mb = 0.0
        if self._gpu_available and self.config.track_gpu_memory:
            gpu_memory_mb = self._get_gpu_memory()
            
        # Disk I/O (if enabled)
        disk_read_mb = disk_write_mb = 0.0
        if self.config.track_disk_io:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_read_mb = disk_io.read_bytes / (1024 * 1024)
                disk_write_mb = disk_io.write_bytes / (1024 * 1024)
                
        # Network I/O (if enabled)
        network_sent_mb = network_recv_mb = 0.0
        if self.config.track_network_io:
            net_io = psutil.net_io_counters()
            if net_io:
                network_sent_mb = net_io.bytes_sent / (1024 * 1024)
                network_recv_mb = net_io.bytes_recv / (1024 * 1024)
                
        return PerformanceSnapshot(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
        
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 * 1024)
        except:
            return 0.0
            
    @contextmanager
    def profile_operation(self, operation_name: str, **context):
        """Context manager for profiling a specific operation"""
        start_time = time.time()
        start_snapshot = self._take_snapshot()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_snapshot = self._take_snapshot()
            
            # Record latency
            latency_ms = (end_time - start_time) * 1000
            measurement = LatencyMeasurement(
                operation_name=operation_name,
                latency_ms=latency_ms,
                timestamp=start_time,
                context=context
            )
            self.latency_measurements[operation_name].append(measurement)
            
            # Update operation count
            self._operation_counts[operation_name] += 1
            
    def measure_throughput(
        self,
        operation_func: Callable,
        operation_name: str,
        duration_seconds: float = 60.0,
        max_operations: int = 1000
    ) -> ThroughputMeasurement:
        """
        Measure throughput for a given operation
        
        Args:
            operation_func: Function to execute repeatedly
            operation_name: Name for the operation
            duration_seconds: How long to run the test
            max_operations: Maximum number of operations to run
            
        Returns:
            ThroughputMeasurement with results
        """
        print(f"Starting throughput measurement for {operation_name}")
        print(f"Duration: {duration_seconds}s, Max operations: {max_operations}")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        operations_count = 0
        
        # Run operations
        while time.time() < end_time and operations_count < max_operations:
            with self.profile_operation(f"{operation_name}_throughput"):
                operation_func()
            operations_count += 1
            
        actual_end_time = time.time()
        actual_duration = actual_end_time - start_time
        throughput = operations_count / actual_duration
        
        measurement = ThroughputMeasurement(
            window_start=start_time,
            window_end=actual_end_time,
            operations_count=operations_count,
            throughput_ops_per_sec=throughput,
            operation_type=operation_name
        )
        
        self.throughput_measurements.append(measurement)
        
        print(f"Throughput measurement complete:")
        print(f"  Operations: {operations_count}")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} ops/sec")
        
        return measurement
        
    def get_latency_statistics(self, operation_name: str) -> Dict[str, float]:
        """Get latency statistics for a specific operation"""
        if operation_name not in self.latency_measurements:
            return {}
            
        latencies = [m.latency_ms for m in self.latency_measurements[operation_name]]
        
        if not latencies:
            return {}
            
        return {
            "count": len(latencies),
            "mean_ms": np.mean(latencies),
            "median_ms": np.median(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p90_ms": np.percentile(latencies, 90),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99)
        }
        
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get resource usage statistics from monitoring snapshots"""
        if not self.snapshots:
            return {}
            
        snapshots_list = list(self.snapshots)
        
        # CPU statistics
        cpu_values = [s.cpu_percent for s in snapshots_list]
        cpu_stats = {
            "mean": np.mean(cpu_values),
            "max": np.max(cpu_values),
            "min": np.min(cpu_values),
            "std": np.std(cpu_values)
        }
        
        # Memory statistics
        memory_values = [s.memory_mb for s in snapshots_list]
        memory_stats = {
            "mean_mb": np.mean(memory_values),
            "max_mb": np.max(memory_values),
            "min_mb": np.min(memory_values),
            "std_mb": np.std(memory_values)
        }
        
        # GPU memory statistics (if available)
        gpu_memory_stats = {}
        if self._gpu_available:
            gpu_memory_values = [s.gpu_memory_mb for s in snapshots_list if s.gpu_memory_mb > 0]
            if gpu_memory_values:
                gpu_memory_stats = {
                    "mean_mb": np.mean(gpu_memory_values),
                    "max_mb": np.max(gpu_memory_values),
                    "min_mb": np.min(gpu_memory_values),
                    "std_mb": np.std(gpu_memory_values)
                }
                
        return {
            "cpu": cpu_stats,
            "memory": memory_stats,
            "gpu_memory": gpu_memory_stats,
            "monitoring_duration_seconds": snapshots_list[-1].timestamp - snapshots_list[0].timestamp if snapshots_list else 0,
            "total_samples": len(snapshots_list)
        }
        
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Analyze performance data and identify potential issues"""
        issues = []
        
        # Check for high CPU usage
        resource_stats = self.get_resource_statistics()
        if resource_stats.get("cpu", {}).get("mean", 0) > 80:
            issues.append({
                "type": "high_cpu_usage",
                "severity": "warning",
                "description": f"Average CPU usage is {resource_stats['cpu']['mean']:.1f}%",
                "recommendation": "Consider optimizing CPU-intensive operations or scaling horizontally"
            })
            
        # Check for memory growth
        if self.snapshots and len(self.snapshots) > 100:
            snapshots_list = list(self.snapshots)
            early_memory = np.mean([s.memory_mb for s in snapshots_list[:50]])
            late_memory = np.mean([s.memory_mb for s in snapshots_list[-50:]])
            memory_growth = late_memory - early_memory
            
            if memory_growth > 500:  # 500MB growth
                issues.append({
                    "type": "potential_memory_leak",
                    "severity": "error",
                    "description": f"Memory usage increased by {memory_growth:.1f}MB during monitoring",
                    "recommendation": "Check for memory leaks in long-running operations"
                })
                
        # Check for high latency operations
        for operation_name, measurements in self.latency_measurements.items():
            if len(measurements) > 10:
                stats = self.get_latency_statistics(operation_name)
                if stats.get("p95_ms", 0) > 1000:  # 1 second P95
                    issues.append({
                        "type": "high_latency",
                        "severity": "warning", 
                        "description": f"Operation '{operation_name}' has P95 latency of {stats['p95_ms']:.1f}ms",
                        "recommendation": "Optimize the operation or consider caching/async processing"
                    })
                    
        return issues
        
    def generate_performance_report(self, output_dir: str = "performance_reports") -> str:
        """Generate comprehensive performance analysis report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, f"performance_report_{int(time.time())}.md")
        
        with open(report_path, 'w') as f:
            f.write("# Performance Analysis Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resource usage summary
            f.write("## Resource Usage Summary\n\n")
            resource_stats = self.get_resource_statistics()
            
            if resource_stats:
                f.write("### CPU Usage\n")
                cpu_stats = resource_stats.get("cpu", {})
                f.write(f"- Mean: {cpu_stats.get('mean', 0):.2f}%\n")
                f.write(f"- Maximum: {cpu_stats.get('max', 0):.2f}%\n")
                f.write(f"- Standard Deviation: {cpu_stats.get('std', 0):.2f}%\n\n")
                
                f.write("### Memory Usage\n")
                memory_stats = resource_stats.get("memory", {})
                f.write(f"- Mean: {memory_stats.get('mean_mb', 0):.1f} MB\n")
                f.write(f"- Maximum: {memory_stats.get('max_mb', 0):.1f} MB\n")
                f.write(f"- Standard Deviation: {memory_stats.get('std_mb', 0):.1f} MB\n\n")
                
                if resource_stats.get("gpu_memory"):
                    f.write("### GPU Memory Usage\n")
                    gpu_stats = resource_stats["gpu_memory"]
                    f.write(f"- Mean: {gpu_stats.get('mean_mb', 0):.1f} MB\n")
                    f.write(f"- Maximum: {gpu_stats.get('max_mb', 0):.1f} MB\n")
                    f.write(f"- Standard Deviation: {gpu_stats.get('std_mb', 0):.1f} MB\n\n")
                    
            # Latency analysis
            f.write("## Latency Analysis\n\n")
            if self.latency_measurements:
                f.write("| Operation | Count | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |\n")
                f.write("|-----------|-------|-----------|----------|----------|----------|\n")
                
                for operation_name in sorted(self.latency_measurements.keys()):
                    stats = self.get_latency_statistics(operation_name)
                    if stats:
                        f.write(f"| {operation_name} | {stats['count']} | {stats['mean_ms']:.2f} | "
                               f"{stats['p50_ms']:.2f} | {stats['p95_ms']:.2f} | {stats['p99_ms']:.2f} |\n")
            else:
                f.write("No latency measurements recorded.\n")
                
            f.write("\n")
            
            # Throughput analysis
            f.write("## Throughput Analysis\n\n")
            if self.throughput_measurements:
                f.write("| Operation | Duration (s) | Operations | Throughput (ops/s) |\n")
                f.write("|-----------|--------------|------------|--------------------|\n")
                
                for measurement in self.throughput_measurements:
                    duration = measurement.window_end - measurement.window_start
                    f.write(f"| {measurement.operation_type} | {duration:.2f} | "
                           f"{measurement.operations_count} | {measurement.throughput_ops_per_sec:.2f} |\n")
            else:
                f.write("No throughput measurements recorded.\n")
                
            f.write("\n")
            
            # Performance issues
            f.write("## Detected Issues\n\n")
            issues = self.detect_performance_issues()
            if issues:
                for issue in issues:
                    f.write(f"### {issue['type'].replace('_', ' ').title()} ({issue['severity'].upper()})\n")
                    f.write(f"{issue['description']}\n\n")
                    f.write(f"**Recommendation:** {issue['recommendation']}\n\n")
            else:
                f.write("No performance issues detected.\n\n")
                
            # Methodology
            f.write("## Methodology\n\n")
            f.write(f"- Sampling interval: {self.config.sample_interval}s\n")
            f.write(f"- Maximum samples: {self.config.max_samples}\n")
            f.write(f"- GPU monitoring: {'Enabled' if self._gpu_available else 'Disabled'}\n")
            f.write(f"- Total monitoring duration: {resource_stats.get('monitoring_duration_seconds', 0):.1f}s\n")
            
        print(f"Performance report generated: {report_path}")
        return report_path
        
    def create_visualizations(self, output_dir: str = "performance_plots"):
        """Create performance visualization plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.snapshots:
            print("No monitoring data available for visualization")
            return
            
        snapshots_list = list(self.snapshots)
        timestamps = [s.timestamp for s in snapshots_list]
        start_time = timestamps[0]
        relative_times = [(t - start_time) / 60 for t in timestamps]  # Convert to minutes
        
        # CPU and Memory usage over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        cpu_values = [s.cpu_percent for s in snapshots_list]
        plt.plot(relative_times, cpu_values, label='CPU Usage (%)', color='red')
        plt.title('CPU Usage Over Time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('CPU Usage (%)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        memory_values = [s.memory_mb / 1024 for s in snapshots_list]  # Convert to GB
        plt.plot(relative_times, memory_values, label='Memory Usage (GB)', color='blue')
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Memory Usage (GB)')
        plt.grid(True, alpha=0.3)
        
        # GPU memory (if available)
        if self._gpu_available:
            plt.subplot(2, 2, 3)
            gpu_memory_values = [s.gpu_memory_mb / 1024 for s in snapshots_list]
            plt.plot(relative_times, gpu_memory_values, label='GPU Memory (GB)', color='green')
            plt.title('GPU Memory Usage Over Time')
            plt.xlabel('Time (minutes)')
            plt.ylabel('GPU Memory (GB)')
            plt.grid(True, alpha=0.3)
            
        # Latency distribution (if available)
        if self.latency_measurements:
            plt.subplot(2, 2, 4)
            all_latencies = []
            labels = []
            
            for operation_name, measurements in list(self.latency_measurements.items())[:3]:  # Top 3 operations
                latencies = [m.latency_ms for m in measurements]
                all_latencies.extend(latencies)
                labels.extend([operation_name] * len(latencies))
                
            if all_latencies:
                plt.hist([self.latency_measurements[op] for op in list(self.latency_measurements.keys())[:3]], 
                        bins=30, alpha=0.7, label=list(self.latency_measurements.keys())[:3])
                plt.title('Latency Distribution')
                plt.xlabel('Latency (ms)')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance visualizations saved to {output_dir}")


# Example usage and testing
def example_profiler_usage():
    """Demonstrate profiler usage with example workload"""
    
    def mock_inference_operation():
        """Mock inference operation for testing"""
        # Simulate some work
        time.sleep(np.random.uniform(0.05, 0.2))
        
    def mock_training_operation():
        """Mock training operation for testing"""
        time.sleep(np.random.uniform(0.1, 0.5))
        
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Start monitoring
    profiler.start_monitoring()
    
    print("Running example workload...")
    
    # Simulate some operations with profiling
    for i in range(10):
        with profiler.profile_operation("inference", batch_size=8):
            mock_inference_operation()
            
        if i % 3 == 0:
            with profiler.profile_operation("training_step", epoch=i//3):
                mock_training_operation()
                
    # Measure throughput
    print("Measuring inference throughput...")
    profiler.measure_throughput(
        mock_inference_operation,
        "inference_throughput",
        duration_seconds=10.0
    )
    
    # Wait a bit more for monitoring data
    time.sleep(2)
    
    # Stop monitoring
    profiler.stop_monitoring()
    
    # Generate reports
    report_path = profiler.generate_performance_report()
    profiler.create_visualizations()
    
    # Show some statistics
    print("\nLatency Statistics:")
    for operation in profiler.latency_measurements.keys():
        stats = profiler.get_latency_statistics(operation)
        print(f"  {operation}: {stats}")
        
    print("\nResource Statistics:")
    print(profiler.get_resource_statistics())
    
    print("\nDetected Issues:")
    issues = profiler.detect_performance_issues()
    for issue in issues:
        print(f"  {issue['type']}: {issue['description']}")
        
    return profiler


if __name__ == "__main__":
    example_profiler_usage()