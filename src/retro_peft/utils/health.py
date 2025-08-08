"""
Health checks and system diagnostics for retro-peft-adapters.

Provides comprehensive health monitoring for all system components
including models, retrievers, databases, and infrastructure.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from .logging import get_global_logger
from .monitoring import HealthCheck, get_health_monitor


def check_system_resources() -> HealthCheck:
    """Check system resource availability"""
    try:
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory check
        memory = psutil.virtual_memory()

        # Disk check
        disk = psutil.disk_usage("/")

        details = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage_percent": (disk.used / disk.total) * 100,
            "disk_free_gb": disk.free / (1024**3),
        }

        # Determine status
        if cpu_percent > 90 or memory.percent > 90 or (disk.used / disk.total) > 0.95:
            status = "unhealthy"
            message = "Critical resource usage detected"
        elif cpu_percent > 80 or memory.percent > 80 or (disk.used / disk.total) > 0.85:
            status = "degraded"
            message = "High resource usage detected"
        else:
            status = "healthy"
            message = "System resources within normal limits"

        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            timestamp=time.time(),
            details=details,
        )

    except Exception as e:
        return HealthCheck(
            name="system_resources",
            status="unhealthy",
            message=f"Failed to check system resources: {e}",
            timestamp=time.time(),
            details={"error": str(e)},
        )


def check_gpu_availability() -> HealthCheck:
    """Check GPU availability and status"""
    try:
        import torch

        if not torch.cuda.is_available():
            return HealthCheck(
                name="gpu_availability",
                status="healthy",
                message="CUDA not available (CPU-only mode)",
                timestamp=time.time(),
                details={"cuda_available": False, "device_count": 0},
            )

        device_count = torch.cuda.device_count()
        devices = []

        for i in range(device_count):
            device_props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = device_props.total_memory

            devices.append(
                {
                    "device_id": i,
                    "name": device_props.name,
                    "memory_total_gb": memory_total / (1024**3),
                    "memory_allocated_gb": memory_allocated / (1024**3),
                    "memory_reserved_gb": memory_reserved / (1024**3),
                    "memory_free_gb": (memory_total - memory_reserved) / (1024**3),
                    "memory_usage_percent": (memory_reserved / memory_total) * 100,
                }
            )

        # Check for memory issues
        high_usage_devices = [d for d in devices if d["memory_usage_percent"] > 90]

        if high_usage_devices:
            status = "degraded"
            message = f"High GPU memory usage on {len(high_usage_devices)} device(s)"
        else:
            status = "healthy"
            message = f"{device_count} GPU device(s) available and healthy"

        return HealthCheck(
            name="gpu_availability",
            status=status,
            message=message,
            timestamp=time.time(),
            details={"cuda_available": True, "device_count": device_count, "devices": devices},
        )

    except ImportError:
        return HealthCheck(
            name="gpu_availability",
            status="healthy",
            message="PyTorch not available (CPU-only mode)",
            timestamp=time.time(),
            details={"torch_available": False},
        )
    except Exception as e:
        return HealthCheck(
            name="gpu_availability",
            status="unhealthy",
            message=f"GPU check failed: {e}",
            timestamp=time.time(),
            details={"error": str(e)},
        )


def check_disk_space() -> HealthCheck:
    """Check disk space for critical directories"""
    try:
        directories_to_check = [
            "/",  # Root directory
            "/tmp",  # Temporary directory
            os.path.expanduser("~"),  # Home directory
        ]

        # Add current working directory
        directories_to_check.append(os.getcwd())

        disk_info = {}
        critical_issues = []
        warnings = []

        for directory in directories_to_check:
            if os.path.exists(directory):
                usage = psutil.disk_usage(directory)
                usage_percent = (usage.used / usage.total) * 100
                free_gb = usage.free / (1024**3)

                disk_info[directory] = {
                    "total_gb": usage.total / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "free_gb": free_gb,
                    "usage_percent": usage_percent,
                }

                if usage_percent > 95 or free_gb < 1:
                    critical_issues.append(
                        f"{directory}: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
                    )
                elif usage_percent > 85 or free_gb < 5:
                    warnings.append(f"{directory}: {usage_percent:.1f}% used, {free_gb:.1f}GB free")

        if critical_issues:
            status = "unhealthy"
            message = f"Critical disk space issues: {'; '.join(critical_issues)}"
        elif warnings:
            status = "degraded"
            message = f"Disk space warnings: {'; '.join(warnings)}"
        else:
            status = "healthy"
            message = "Disk space levels are healthy"

        return HealthCheck(
            name="disk_space",
            status=status,
            message=message,
            timestamp=time.time(),
            details=disk_info,
        )

    except Exception as e:
        return HealthCheck(
            name="disk_space",
            status="unhealthy",
            message=f"Disk space check failed: {e}",
            timestamp=time.time(),
            details={"error": str(e)},
        )


def check_file_permissions() -> HealthCheck:
    """Check file system permissions for critical operations"""
    try:
        issues = []

        # Test write permissions in temp directory
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                tmp_file.write(b"test")
                tmp_file.flush()
        except Exception as e:
            issues.append(f"Cannot write to temp directory: {e}")

        # Test write permissions in current directory
        try:
            test_file = "test_permissions.tmp"
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            issues.append(f"Cannot write to current directory: {e}")

        # Test read permissions for common config locations
        config_paths = [os.path.expanduser("~/.config"), "/etc", os.getcwd()]

        for path in config_paths:
            if os.path.exists(path) and not os.access(path, os.R_OK):
                issues.append(f"Cannot read from {path}")

        if issues:
            status = "unhealthy"
            message = f"File permission issues: {'; '.join(issues)}"
        else:
            status = "healthy"
            message = "File permissions are adequate"

        return HealthCheck(
            name="file_permissions",
            status=status,
            message=message,
            timestamp=time.time(),
            details={"issues": issues},
        )

    except Exception as e:
        return HealthCheck(
            name="file_permissions",
            status="unhealthy",
            message=f"Permission check failed: {e}",
            timestamp=time.time(),
            details={"error": str(e)},
        )


def check_python_environment() -> HealthCheck:
    """Check Python environment and dependencies"""
    try:
        import sys

        details = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": sys.platform,
            "path": sys.path[:5],  # First 5 entries
        }

        # Check critical dependencies
        dependencies = {
            "torch": None,
            "transformers": None,
            "sentence_transformers": None,
            "numpy": None,
            "faiss": None,
            "psutil": None,
        }

        missing_deps = []
        version_issues = []

        for dep_name in dependencies.keys():
            try:
                if dep_name == "faiss":
                    # Try both faiss-cpu and faiss-gpu
                    try:
                        import faiss

                        dependencies[dep_name] = getattr(faiss, "__version__", "unknown")
                    except ImportError:
                        try:
                            import faiss_cpu as faiss

                            dependencies[dep_name] = getattr(faiss, "__version__", "unknown")
                        except ImportError:
                            missing_deps.append(dep_name)
                else:
                    module = __import__(dep_name)
                    dependencies[dep_name] = getattr(module, "__version__", "unknown")
            except ImportError:
                if dep_name in ["torch", "transformers", "numpy"]:
                    missing_deps.append(dep_name)  # Critical dependencies
                else:
                    dependencies[dep_name] = "not_installed"

        details["dependencies"] = dependencies

        if missing_deps:
            status = "unhealthy"
            message = f"Critical dependencies missing: {', '.join(missing_deps)}"
        elif version_issues:
            status = "degraded"
            message = f"Dependency version issues: {', '.join(version_issues)}"
        else:
            status = "healthy"
            message = "Python environment is healthy"

        return HealthCheck(
            name="python_environment",
            status=status,
            message=message,
            timestamp=time.time(),
            details=details,
        )

    except Exception as e:
        return HealthCheck(
            name="python_environment",
            status="unhealthy",
            message=f"Environment check failed: {e}",
            timestamp=time.time(),
            details={"error": str(e)},
        )


def check_network_connectivity() -> HealthCheck:
    """Check network connectivity for external services"""
    try:
        import socket
        import urllib.request

        connectivity_tests = [
            ("DNS Resolution", "google.com", lambda: socket.gethostbyname("google.com")),
            (
                "HTTP Connectivity",
                "http://httpbin.org/status/200",
                lambda: urllib.request.urlopen("http://httpbin.org/status/200", timeout=5),
            ),
            (
                "HTTPS Connectivity",
                "https://httpbin.org/status/200",
                lambda: urllib.request.urlopen("https://httpbin.org/status/200", timeout=5),
            ),
        ]

        results = {}
        failures = []

        for test_name, target, test_func in connectivity_tests:
            try:
                start_time = time.time()
                test_func()
                duration = (time.time() - start_time) * 1000

                results[test_name] = {
                    "status": "success",
                    "target": target,
                    "duration_ms": duration,
                }
            except Exception as e:
                results[test_name] = {"status": "failed", "target": target, "error": str(e)}
                failures.append(f"{test_name} to {target}")

        if len(failures) >= len(connectivity_tests):
            status = "unhealthy"
            message = "No network connectivity"
        elif failures:
            status = "degraded"
            message = f"Partial network connectivity: {', '.join(failures)} failed"
        else:
            status = "healthy"
            message = "Network connectivity is healthy"

        return HealthCheck(
            name="network_connectivity",
            status=status,
            message=message,
            timestamp=time.time(),
            details=results,
        )

    except Exception as e:
        return HealthCheck(
            name="network_connectivity",
            status="unhealthy",
            message=f"Network check failed: {e}",
            timestamp=time.time(),
            details={"error": str(e)},
        )


def check_model_loading() -> HealthCheck:
    """Check if models can be loaded successfully"""
    try:
        test_results = {}

        # Test small model loading
        try:
            from transformers import AutoModel, AutoTokenizer

            # Try to load a very small model for testing
            model_name = "prajjwal1/bert-tiny"  # Very small model

            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            load_time = (time.time() - start_time) * 1000

            # Test basic inference
            inputs = tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            test_results["model_loading"] = {
                "status": "success",
                "model": model_name,
                "load_time_ms": load_time,
                "output_shape": list(outputs.last_hidden_state.shape),
            }

        except Exception as e:
            test_results["model_loading"] = {"status": "failed", "error": str(e)}

        # Test sentence transformer loading
        try:
            from sentence_transformers import SentenceTransformer

            model_name = "all-MiniLM-L6-v2"
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_time = (time.time() - start_time) * 1000

            # Test encoding
            embeddings = model.encode(["Hello world"])

            test_results["sentence_transformer"] = {
                "status": "success",
                "model": model_name,
                "load_time_ms": load_time,
                "embedding_dim": embeddings.shape[1],
            }

        except Exception as e:
            test_results["sentence_transformer"] = {"status": "failed", "error": str(e)}

        # Determine overall status
        successes = sum(1 for result in test_results.values() if result["status"] == "success")
        total_tests = len(test_results)

        if successes == 0:
            status = "unhealthy"
            message = "Cannot load any models"
        elif successes < total_tests:
            status = "degraded"
            message = f"Some model loading issues ({successes}/{total_tests} successful)"
        else:
            status = "healthy"
            message = "Model loading is healthy"

        return HealthCheck(
            name="model_loading",
            status=status,
            message=message,
            timestamp=time.time(),
            details=test_results,
        )

    except Exception as e:
        return HealthCheck(
            name="model_loading",
            status="unhealthy",
            message=f"Model loading check failed: {e}",
            timestamp=time.time(),
            details={"error": str(e)},
        )


def register_default_health_checks():
    """Register all default health checks"""
    health_monitor = get_health_monitor()
    logger = get_global_logger()

    health_checks = [
        ("system_resources", check_system_resources),
        ("gpu_availability", check_gpu_availability),
        ("disk_space", check_disk_space),
        ("file_permissions", check_file_permissions),
        ("python_environment", check_python_environment),
        ("network_connectivity", check_network_connectivity),
        # Note: model_loading check is resource-intensive, register manually if needed
    ]

    for name, check_func in health_checks:
        health_monitor.register_check(name, check_func)

    logger.info(f"Registered {len(health_checks)} default health checks")


def run_system_diagnostics() -> Dict[str, Any]:
    """
    Run comprehensive system diagnostics.

    Returns:
        Comprehensive diagnostic report
    """
    logger = get_global_logger()
    logger.info("Running system diagnostics...")

    # Register default checks if not already done
    register_default_health_checks()

    # Get health monitor
    health_monitor = get_health_monitor()

    # Run all health checks
    health_results = health_monitor.get_overall_health()

    # Additional diagnostic information
    diagnostic_info = {
        "timestamp": time.time(),
        "system_info": {
            "platform": os.uname() if hasattr(os, "uname") else "unknown",
            "python_version": os.sys.version,
            "working_directory": os.getcwd(),
            "environment_variables": {
                key: value
                for key, value in os.environ.items()
                if any(
                    keyword in key.upper()
                    for keyword in ["CUDA", "PATH", "PYTHON", "HOME", "USER", "LANG"]
                )
            },
        },
        "process_info": {
            "pid": os.getpid(),
            "ppid": os.getppid() if hasattr(os, "getppid") else None,
            "threads": threading.active_count(),
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        },
        "health_status": health_results,
    }

    logger.info(f"System diagnostics completed. Overall status: {health_results['status']}")

    return diagnostic_info


def create_diagnostic_report(output_file: Optional[str] = None) -> str:
    """
    Create a comprehensive diagnostic report.

    Args:
        output_file: Optional file to save the report

    Returns:
        Diagnostic report as string
    """
    import json
    from datetime import datetime

    # Run diagnostics
    diagnostics = run_system_diagnostics()

    # Format report
    report_lines = [
        "=" * 80,
        "RETRO-PEFT-ADAPTERS SYSTEM DIAGNOSTIC REPORT",
        "=" * 80,
        f"Generated: {datetime.fromtimestamp(diagnostics['timestamp']).isoformat()}",
        f"Overall Health: {diagnostics['health_status']['status'].upper()}",
        "",
        f"Health Summary: {diagnostics['health_status']['message']}",
        "",
        "HEALTH CHECK DETAILS:",
        "-" * 40,
    ]

    for check_name, check_result in diagnostics["health_status"]["checks"].items():
        report_lines.extend(
            [
                f"{check_name}:",
                f"  Status: {check_result['status']}",
                f"  Message: {check_result['message']}",
                "",
            ]
        )

    report_lines.extend(
        [
            "SYSTEM INFORMATION:",
            "-" * 40,
            f"Platform: {diagnostics['system_info']['platform']}",
            f"Python: {diagnostics['system_info']['python_version']}",
            f"Working Directory: {diagnostics['system_info']['working_directory']}",
            f"Process ID: {diagnostics['process_info']['pid']}",
            f"Memory Usage: {diagnostics['process_info']['memory_mb']:.1f} MB",
            f"Active Threads: {diagnostics['process_info']['threads']}",
            "",
            "ENVIRONMENT VARIABLES:",
            "-" * 40,
        ]
    )

    for key, value in diagnostics["system_info"]["environment_variables"].items():
        report_lines.append(f"{key}: {value}")

    report_lines.extend(
        [
            "",
            "RAW DIAGNOSTIC DATA:",
            "-" * 40,
            json.dumps(diagnostics, indent=2, default=str),
            "",
            "=" * 80,
        ]
    )

    report_text = "\n".join(report_lines)

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)

        logger = get_global_logger()
        logger.info(f"Diagnostic report saved to: {output_file}")

    return report_text
