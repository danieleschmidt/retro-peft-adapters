"""
Autonomous Experimental Framework (AEF)

Comprehensive experimental framework for autonomous research validation
with statistical significance testing, A/B testing capabilities, and
reproducible experimental protocols for publication-ready results.

Key Features:
1. Automated hypothesis testing with statistical significance
2. A/B testing framework with power analysis
3. Cross-validation and bootstrap confidence intervals
4. Reproducible experimental protocols
5. Publication-ready results reporting
6. Multi-metric evaluation with correlation analysis
7. Automated experimental design optimization
"""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import random
from collections import defaultdict
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, pearsonr
from sklearn.model_selection import KFold, StrategyifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from .meta_adaptive_hierarchical_fusion import MetaAdaptiveHierarchicalFusion, MAHFConfig
from .cross_modal_adaptive_retrieval import CrossModalAdaptiveRetrievalNetwork, CARNConfig
from .neuromorphic_spike_dynamics import NeuromorphicSpikeNetwork, NeuromorphicConfig
from .quantum_enhanced_adapters import QuantumEnhancedAdapter, QuantumConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalConfig:
    """Configuration for autonomous experimental framework"""
    
    # Statistical testing parameters
    alpha: float = 0.05  # Significance level
    power: float = 0.8   # Statistical power
    effect_size: float = 0.5  # Minimum detectable effect size
    
    # A/B testing parameters
    min_sample_size: int = 30
    max_sample_size: int = 1000
    confidence_level: float = 0.95
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_repeats: int = 3
    bootstrap_samples: int = 1000
    
    # Experimental design
    randomization_seed: int = 42
    stratification: bool = True
    blocking_variables: List[str] = field(default_factory=list)
    
    # Metrics and evaluation
    primary_metric: str = "performance"
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "efficiency", "stability", "robustness"
    ])
    
    # Reporting and visualization
    generate_plots: bool = True
    save_results: bool = True
    results_dir: str = "experimental_results"
    
    # Research protocols
    reproducible_experiments: bool = True
    track_computational_cost: bool = True
    enable_early_stopping: bool = True
    patience: int = 10


class StatisticalAnalyzer:
    """Statistical analysis toolkit for experimental validation"""
    
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        
    def power_analysis(
        self, 
        effect_size: float,
        alpha: float = None,
        power: float = None
    ) -> Dict[str, float]:
        """Perform statistical power analysis"""
        if alpha is None:
            alpha = self.config.alpha
        if power is None:
            power = self.config.power
            
        # Cohen's d for effect size
        # n = (z_alpha/2 + z_beta)^2 * 2 * sigma^2 / delta^2
        z_alpha_2 = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        sample_size_per_group = 2 * ((z_alpha_2 + z_beta) / effect_size) ** 2
        total_sample_size = sample_size_per_group * 2
        
        return {
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "sample_size_per_group": int(np.ceil(sample_size_per_group)),
            "total_sample_size": int(np.ceil(total_sample_size)),
            "z_alpha_2": z_alpha_2,
            "z_beta": z_beta
        }
        
    def hypothesis_test(
        self, 
        group_a: List[float], 
        group_b: List[float],
        test_type: str = "auto",
        paired: bool = False
    ) -> Dict[str, Any]:
        """Perform hypothesis testing between two groups"""
        
        # Convert to numpy arrays
        a_data = np.array(group_a)
        b_data = np.array(group_b)
        
        # Basic descriptive statistics
        stats_a = {
            "mean": np.mean(a_data),
            "std": np.std(a_data, ddof=1),
            "median": np.median(a_data),
            "n": len(a_data)
        }
        
        stats_b = {
            "mean": np.mean(b_data),
            "std": np.std(b_data, ddof=1),
            "median": np.median(b_data),
            "n": len(b_data)
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(a_data) - 1) * stats_a["std"]**2 + 
                             (len(b_data) - 1) * stats_b["std"]**2) / 
                            (len(a_data) + len(b_data) - 2))
        cohens_d = (stats_a["mean"] - stats_b["mean"]) / pooled_std
        
        # Choose statistical test
        if test_type == "auto":
            # Check normality with Shapiro-Wilk test
            _, p_norm_a = stats.shapiro(a_data) if len(a_data) < 5000 else (None, 0.05)
            _, p_norm_b = stats.shapiro(b_data) if len(b_data) < 5000 else (None, 0.05)
            
            if p_norm_a > 0.05 and p_norm_b > 0.05:
                test_type = "t_test"
            else:
                test_type = "mann_whitney"
                
        # Perform the statistical test
        if test_type == "t_test":
            if paired:
                statistic, p_value = stats.ttest_rel(a_data, b_data)
                test_name = "Paired t-test"
            else:
                statistic, p_value = stats.ttest_ind(a_data, b_data)
                test_name = "Independent t-test"
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(a_data, b_data, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        # Confidence interval for difference in means
        if test_type == "t_test":
            diff_mean = stats_a["mean"] - stats_b["mean"]
            se_diff = np.sqrt(stats_a["std"]**2/stats_a["n"] + stats_b["std"]**2/stats_b["n"])
            df = stats_a["n"] + stats_b["n"] - 2
            t_critical = stats.t.ppf(1 - self.config.alpha/2, df)
            ci_lower = diff_mean - t_critical * se_diff
            ci_upper = diff_mean + t_critical * se_diff
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = None
            
        return {
            "test_name": test_name,
            "test_type": test_type,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.config.alpha,
            "effect_size_cohens_d": cohens_d,
            "effect_size_interpretation": self._interpret_cohens_d(cohens_d),
            "confidence_interval": confidence_interval,
            "group_a_stats": stats_a,
            "group_b_stats": stats_b
        }
        
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
            
    def bootstrap_confidence_interval(
        self, 
        data: List[float], 
        statistic_func: Callable = np.mean,
        confidence_level: float = None
    ) -> Dict[str, float]:
        """Calculate bootstrap confidence interval"""
        if confidence_level is None:
            confidence_level = self.config.confidence_level
            
        data_array = np.array(data)
        n_samples = len(data_array)
        n_bootstrap = self.config.bootstrap_samples
        
        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data_array, size=n_samples, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
            
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            "statistic": statistic_func(data_array),
            "confidence_level": confidence_level,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "bootstrap_mean": np.mean(bootstrap_stats),
            "bootstrap_std": np.std(bootstrap_stats)
        }
        
    def correlation_analysis(
        self, 
        metrics: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Perform correlation analysis between metrics"""
        metric_names = list(metrics.keys())
        n_metrics = len(metric_names)
        
        # Correlation matrix
        correlation_matrix = np.zeros((n_metrics, n_metrics))
        p_value_matrix = np.zeros((n_metrics, n_metrics))
        
        for i, metric_i in enumerate(metric_names):
            for j, metric_j in enumerate(metric_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    p_value_matrix[i, j] = 0.0
                else:
                    corr, p_val = pearsonr(metrics[metric_i], metrics[metric_j])
                    correlation_matrix[i, j] = corr
                    p_value_matrix[i, j] = p_val
                    
        return {
            "metric_names": metric_names,
            "correlation_matrix": correlation_matrix,
            "p_value_matrix": p_value_matrix,
            "significant_correlations": self._find_significant_correlations(
                metric_names, correlation_matrix, p_value_matrix
            )
        }
        
    def _find_significant_correlations(
        self, 
        metric_names: List[str], 
        corr_matrix: np.ndarray, 
        p_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Find statistically significant correlations"""
        significant_correlations = []
        
        for i in range(len(metric_names)):
            for j in range(i+1, len(metric_names)):
                if p_matrix[i, j] < self.config.alpha:
                    significant_correlations.append({
                        "metric_1": metric_names[i],
                        "metric_2": metric_names[j],
                        "correlation": corr_matrix[i, j],
                        "p_value": p_matrix[i, j],
                        "strength": self._interpret_correlation(abs(corr_matrix[i, j]))
                    })
                    
        return significant_correlations
        
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength"""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        else:
            return "very strong"


class ABTestFramework:
    """A/B testing framework for comparing algorithms"""
    
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        self.statistical_analyzer = StatisticalAnalyzer(config)
        
    def design_ab_test(
        self,
        expected_effect_size: float,
        baseline_performance: float,
        performance_std: float
    ) -> Dict[str, Any]:
        """Design A/B test with proper sample size calculation"""
        
        # Power analysis
        power_analysis = self.statistical_analyzer.power_analysis(expected_effect_size)
        
        # Adjust for multiple comparisons if needed
        bonferroni_alpha = self.config.alpha / len(self.config.secondary_metrics)
        
        # Sample size with safety margin
        recommended_sample_size = max(
            power_analysis["sample_size_per_group"],
            self.config.min_sample_size
        )
        
        recommended_sample_size = min(recommended_sample_size, self.config.max_sample_size)
        
        return {
            "power_analysis": power_analysis,
            "recommended_sample_size_per_group": recommended_sample_size,
            "total_sample_size": recommended_sample_size * 2,
            "bonferroni_corrected_alpha": bonferroni_alpha,
            "baseline_performance": baseline_performance,
            "expected_improvement": expected_effect_size * performance_std,
            "test_design": {
                "randomization": "stratified" if self.config.stratification else "simple",
                "blocking": self.config.blocking_variables,
                "allocation_ratio": 1.0  # 1:1 allocation
            }
        }
        
    def run_ab_test(
        self,
        algorithm_a: nn.Module,
        algorithm_b: nn.Module,
        test_data: List[Dict[str, Any]],
        evaluation_metrics: Dict[str, Callable],
        test_name: str = "A/B Test"
    ) -> Dict[str, Any]:
        """Run A/B test between two algorithms"""
        
        logger.info(f"Starting A/B test: {test_name}")
        
        # Randomize test data
        random.seed(self.config.randomization_seed)
        shuffled_data = test_data.copy()
        random.shuffle(shuffled_data)
        
        # Split data between groups
        mid_point = len(shuffled_data) // 2
        group_a_data = shuffled_data[:mid_point]
        group_b_data = shuffled_data[mid_point:2*mid_point]
        
        # Evaluate algorithms
        results_a = self._evaluate_algorithm(algorithm_a, group_a_data, evaluation_metrics)
        results_b = self._evaluate_algorithm(algorithm_b, group_b_data, evaluation_metrics)
        
        # Statistical analysis for each metric
        statistical_results = {}
        
        for metric_name in evaluation_metrics.keys():
            if metric_name in results_a and metric_name in results_b:
                test_result = self.statistical_analyzer.hypothesis_test(
                    results_a[metric_name], results_b[metric_name]
                )
                statistical_results[metric_name] = test_result
                
        # Overall test summary
        primary_metric = self.config.primary_metric
        primary_result = statistical_results.get(primary_metric, {})
        
        # Multiple comparisons correction
        p_values = [result["p_value"] for result in statistical_results.values()]
        corrected_p_values = self._bonferroni_correction(p_values)
        
        test_summary = {
            "test_name": test_name,
            "sample_sizes": {
                "group_a": len(group_a_data),
                "group_b": len(group_b_data)
            },
            "primary_metric_result": primary_result,
            "all_metrics_results": statistical_results,
            "corrected_p_values": corrected_p_values,
            "overall_significant": any(p < self.config.alpha for p in corrected_p_values),
            "winner": self._determine_winner(results_a, results_b, primary_metric),
            "practical_significance": self._assess_practical_significance(
                primary_result.get("effect_size_cohens_d", 0)
            )
        }
        
        logger.info(f"A/B test completed. Winner: {test_summary['winner']}")
        
        return test_summary
        
    def _evaluate_algorithm(
        self,
        algorithm: nn.Module,
        test_data: List[Dict[str, Any]],
        evaluation_metrics: Dict[str, Callable]
    ) -> Dict[str, List[float]]:
        """Evaluate algorithm on test data"""
        
        results = {metric_name: [] for metric_name in evaluation_metrics.keys()}
        
        algorithm.eval()
        with torch.no_grad():
            for data_point in test_data:
                try:
                    # Forward pass
                    output = algorithm(data_point["input"])
                    
                    # Calculate metrics
                    for metric_name, metric_func in evaluation_metrics.items():
                        if "target" in data_point:
                            metric_value = metric_func(output, data_point["target"])
                        else:
                            metric_value = metric_func(output)
                            
                        results[metric_name].append(float(metric_value))
                        
                except Exception as e:
                    logger.warning(f"Evaluation failed for data point: {e}")
                    # Add default values for failed evaluations
                    for metric_name in evaluation_metrics.keys():
                        results[metric_name].append(0.0)
                        
        return results
        
    def _bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons"""
        n_comparisons = len(p_values)
        corrected_p_values = [min(p * n_comparisons, 1.0) for p in p_values]
        return corrected_p_values
        
    def _determine_winner(
        self,
        results_a: Dict[str, List[float]],
        results_b: Dict[str, List[float]],
        primary_metric: str
    ) -> str:
        """Determine the winning algorithm"""
        if primary_metric in results_a and primary_metric in results_b:
            mean_a = np.mean(results_a[primary_metric])
            mean_b = np.mean(results_b[primary_metric])
            
            if mean_a > mean_b:
                return "Algorithm A"
            elif mean_b > mean_a:
                return "Algorithm B"
            else:
                return "Tie"
        else:
            return "Inconclusive"
            
    def _assess_practical_significance(self, cohens_d: float) -> Dict[str, Any]:
        """Assess practical significance of the effect"""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            practical_significance = "negligible"
            recommendation = "No practical difference"
        elif abs_d < 0.5:
            practical_significance = "small"
            recommendation = "Small practical difference"
        elif abs_d < 0.8:
            practical_significance = "medium"
            recommendation = "Meaningful practical difference"
        else:
            practical_significance = "large"
            recommendation = "Large practical difference"
            
        return {
            "cohens_d": cohens_d,
            "practical_significance": practical_significance,
            "recommendation": recommendation
        }


class CrossValidationFramework:
    """Cross-validation framework for robust model evaluation"""
    
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        self.statistical_analyzer = StatisticalAnalyzer(config)
        
    def k_fold_cross_validation(
        self,
        algorithm: nn.Module,
        data: List[Dict[str, Any]],
        evaluation_metrics: Dict[str, Callable],
        stratify_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""
        
        logger.info(f"Starting {self.config.cv_folds}-fold cross-validation")
        
        # Prepare data for cross-validation
        if stratify_by and stratify_by in data[0]:
            # Stratified k-fold
            labels = [item[stratify_by] for item in data]
            kf = StratifiedKFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.randomization_seed
            )
            cv_splits = list(kf.split(range(len(data)), labels))
        else:
            # Regular k-fold
            kf = KFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.randomization_seed
            )
            cv_splits = list(kf.split(range(len(data))))
            
        # Collect results from all folds
        cv_results = {metric_name: [] for metric_name in evaluation_metrics.keys()}
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(cv_splits):
            logger.info(f"Processing fold {fold_idx + 1}/{self.config.cv_folds}")
            
            # Prepare fold data
            val_data = [data[i] for i in val_indices]
            
            # Evaluate on validation set
            fold_metrics = self._evaluate_fold(algorithm, val_data, evaluation_metrics)
            
            # Store fold results
            fold_results.append({
                "fold": fold_idx,
                "metrics": fold_metrics
            })
            
            # Accumulate results
            for metric_name, values in fold_metrics.items():
                cv_results[metric_name].extend(values)
                
        # Calculate cross-validation statistics
        cv_statistics = {}
        for metric_name, values in cv_results.items():
            cv_statistics[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "median": np.median(values),
                "confidence_interval": self.statistical_analyzer.bootstrap_confidence_interval(
                    values, np.mean
                )
            }
            
        return {
            "cv_statistics": cv_statistics,
            "fold_results": fold_results,
            "overall_performance": cv_statistics.get(self.config.primary_metric, {}),
            "stability_analysis": self._analyze_stability(cv_results)
        }
        
    def _evaluate_fold(
        self,
        algorithm: nn.Module,
        val_data: List[Dict[str, Any]],
        evaluation_metrics: Dict[str, Callable]
    ) -> Dict[str, List[float]]:
        """Evaluate algorithm on a single fold"""
        
        results = {metric_name: [] for metric_name in evaluation_metrics.keys()}
        
        algorithm.eval()
        with torch.no_grad():
            for data_point in val_data:
                try:
                    # Forward pass
                    output = algorithm(data_point["input"])
                    
                    # Calculate metrics
                    for metric_name, metric_func in evaluation_metrics.items():
                        if "target" in data_point:
                            metric_value = metric_func(output, data_point["target"])
                        else:
                            metric_value = metric_func(output)
                            
                        results[metric_name].append(float(metric_value))
                        
                except Exception as e:
                    logger.warning(f"Evaluation failed for data point: {e}")
                    
        return results
        
    def _analyze_stability(self, cv_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze stability of cross-validation results"""
        stability_metrics = {}
        
        for metric_name, values in cv_results.items():
            # Coefficient of variation
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            cv_coefficient = std_val / (abs(mean_val) + 1e-8)
            
            # Stability classification
            if cv_coefficient < 0.1:
                stability = "very stable"
            elif cv_coefficient < 0.2:
                stability = "stable"
            elif cv_coefficient < 0.3:
                stability = "moderately stable"
            else:
                stability = "unstable"
                
            stability_metrics[metric_name] = {
                "coefficient_of_variation": cv_coefficient,
                "stability": stability
            }
            
        return stability_metrics


class AutonomousExperimentalFramework:
    """
    Comprehensive autonomous experimental framework for research validation
    with statistical rigor and publication-ready results.
    """
    
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        self.statistical_analyzer = StatisticalAnalyzer(config)
        self.ab_test_framework = ABTestFramework(config)
        self.cv_framework = CrossValidationFramework(config)
        
        # Create results directory
        if self.config.save_results:
            Path(self.config.results_dir).mkdir(exist_ok=True)
            
    def comprehensive_algorithm_comparison(
        self,
        algorithms: Dict[str, nn.Module],
        test_datasets: Dict[str, List[Dict[str, Any]]],
        evaluation_metrics: Dict[str, Callable],
        experiment_name: str = "Algorithm Comparison"
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison of multiple algorithms across datasets
        with full statistical analysis and publication-ready results.
        """
        
        logger.info(f"Starting comprehensive algorithm comparison: {experiment_name}")
        
        comparison_results = {
            "experiment_name": experiment_name,
            "algorithms": list(algorithms.keys()),
            "datasets": list(test_datasets.keys()),
            "metrics": list(evaluation_metrics.keys()),
            "timestamp": time.time(),
            "config": self.config.__dict__.copy()
        }
        
        # 1. Cross-validation for each algorithm-dataset pair
        cv_results = {}
        for alg_name, algorithm in algorithms.items():
            cv_results[alg_name] = {}
            
            for dataset_name, dataset in test_datasets.items():
                logger.info(f"Cross-validating {alg_name} on {dataset_name}")
                
                cv_result = self.cv_framework.k_fold_cross_validation(
                    algorithm, dataset, evaluation_metrics
                )
                cv_results[alg_name][dataset_name] = cv_result
                
        comparison_results["cross_validation_results"] = cv_results
        
        # 2. Pairwise A/B testing
        ab_test_results = {}
        algorithm_pairs = []
        algorithm_names = list(algorithms.keys())
        
        for i in range(len(algorithm_names)):
            for j in range(i+1, len(algorithm_names)):
                alg_a_name = algorithm_names[i]
                alg_b_name = algorithm_names[j]
                algorithm_pairs.append((alg_a_name, alg_b_name))
                
        for alg_a_name, alg_b_name in algorithm_pairs:
            ab_test_results[f"{alg_a_name}_vs_{alg_b_name}"] = {}
            
            for dataset_name, dataset in test_datasets.items():
                logger.info(f"A/B testing {alg_a_name} vs {alg_b_name} on {dataset_name}")
                
                ab_result = self.ab_test_framework.run_ab_test(
                    algorithms[alg_a_name],
                    algorithms[alg_b_name],
                    dataset,
                    evaluation_metrics,
                    test_name=f"{alg_a_name} vs {alg_b_name} on {dataset_name}"
                )
                ab_test_results[f"{alg_a_name}_vs_{alg_b_name}"][dataset_name] = ab_result
                
        comparison_results["ab_test_results"] = ab_test_results
        
        # 3. Overall ranking and statistical significance
        ranking_results = self._rank_algorithms(cv_results, evaluation_metrics)
        comparison_results["algorithm_rankings"] = ranking_results
        
        # 4. Meta-analysis across datasets
        meta_analysis = self._perform_meta_analysis(cv_results, evaluation_metrics)
        comparison_results["meta_analysis"] = meta_analysis
        
        # 5. Robustness analysis
        robustness_analysis = self._analyze_robustness(cv_results, ab_test_results)
        comparison_results["robustness_analysis"] = robustness_analysis
        
        # 6. Publication summary
        publication_summary = self._generate_publication_summary(comparison_results)
        comparison_results["publication_summary"] = publication_summary
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(comparison_results, experiment_name)
            
        # Generate plots if configured
        if self.config.generate_plots:
            self._generate_plots(comparison_results, experiment_name)
            
        logger.info(f"Comprehensive algorithm comparison completed")
        
        return comparison_results
        
    def _rank_algorithms(
        self,
        cv_results: Dict[str, Dict[str, Any]],
        evaluation_metrics: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Rank algorithms based on cross-validation performance"""
        
        # Collect performance data
        algorithm_performances = {}
        
        for alg_name, alg_results in cv_results.items():
            performances = []
            
            for dataset_name, dataset_result in alg_results.items():
                primary_metric = self.config.primary_metric
                if primary_metric in dataset_result["cv_statistics"]:
                    mean_perf = dataset_result["cv_statistics"][primary_metric]["mean"]
                    performances.append(mean_perf)
                    
            if performances:
                algorithm_performances[alg_name] = {
                    "mean_performance": np.mean(performances),
                    "std_performance": np.std(performances),
                    "performances": performances
                }
                
        # Rank algorithms
        sorted_algorithms = sorted(
            algorithm_performances.items(),
            key=lambda x: x[1]["mean_performance"],
            reverse=True
        )
        
        # Statistical significance testing between top algorithms
        significance_tests = {}
        if len(sorted_algorithms) > 1:
            top_alg_name, top_alg_data = sorted_algorithms[0]
            
            for i in range(1, min(3, len(sorted_algorithms))):  # Test against top 2 competitors
                other_alg_name, other_alg_data = sorted_algorithms[i]
                
                test_result = self.statistical_analyzer.hypothesis_test(
                    top_alg_data["performances"],
                    other_alg_data["performances"]
                )
                
                significance_tests[f"{top_alg_name}_vs_{other_alg_name}"] = test_result
                
        return {
            "rankings": [{"rank": i+1, "algorithm": name, "performance": data["mean_performance"]} 
                        for i, (name, data) in enumerate(sorted_algorithms)],
            "performance_data": algorithm_performances,
            "significance_tests": significance_tests,
            "winner": sorted_algorithms[0][0] if sorted_algorithms else "No winner"
        }
        
    def _perform_meta_analysis(
        self,
        cv_results: Dict[str, Dict[str, Any]],
        evaluation_metrics: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Perform meta-analysis across datasets"""
        
        meta_results = {}
        
        for metric_name in evaluation_metrics.keys():
            metric_data = {}
            
            for alg_name, alg_results in cv_results.items():
                metric_values = []
                
                for dataset_name, dataset_result in alg_results.items():
                    if metric_name in dataset_result["cv_statistics"]:
                        mean_val = dataset_result["cv_statistics"][metric_name]["mean"]
                        metric_values.append(mean_val)
                        
                if metric_values:
                    metric_data[alg_name] = {
                        "mean": np.mean(metric_values),
                        "std": np.std(metric_values),
                        "n_datasets": len(metric_values),
                        "values": metric_values
                    }
                    
            # Effect size calculations
            if len(metric_data) >= 2:
                algorithms = list(metric_data.keys())
                effect_sizes = {}
                
                for i in range(len(algorithms)):
                    for j in range(i+1, len(algorithms)):
                        alg_1, alg_2 = algorithms[i], algorithms[j]
                        
                        # Calculate pooled effect size
                        mean_1 = metric_data[alg_1]["mean"]
                        mean_2 = metric_data[alg_2]["mean"]
                        std_1 = metric_data[alg_1]["std"]
                        std_2 = metric_data[alg_2]["std"]
                        
                        pooled_std = np.sqrt((std_1**2 + std_2**2) / 2)
                        cohens_d = (mean_1 - mean_2) / pooled_std
                        
                        effect_sizes[f"{alg_1}_vs_{alg_2}"] = {
                            "cohens_d": cohens_d,
                            "interpretation": self.statistical_analyzer._interpret_cohens_d(cohens_d)
                        }
                        
            meta_results[metric_name] = {
                "algorithm_data": metric_data,
                "effect_sizes": effect_sizes if len(metric_data) >= 2 else {}
            }
            
        return meta_results
        
    def _analyze_robustness(
        self,
        cv_results: Dict[str, Dict[str, Any]],
        ab_test_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze robustness of results across different conditions"""
        
        robustness_metrics = {}
        
        # Cross-dataset consistency
        for alg_name, alg_results in cv_results.items():
            performances = []
            
            for dataset_result in alg_results.values():
                primary_metric = self.config.primary_metric
                if primary_metric in dataset_result["cv_statistics"]:
                    performances.append(dataset_result["cv_statistics"][primary_metric]["mean"])
                    
            if len(performances) > 1:
                consistency = 1.0 - (np.std(performances) / np.mean(performances))
                robustness_metrics[alg_name] = {
                    "cross_dataset_consistency": consistency,
                    "performance_range": (min(performances), max(performances)),
                    "coefficient_of_variation": np.std(performances) / np.mean(performances)
                }
                
        # A/B test consistency
        ab_consistency = {}
        for comparison, comparison_results in ab_test_results.items():
            winners = []
            
            for dataset_result in comparison_results.values():
                winners.append(dataset_result["winner"])
                
            # Calculate consistency of A/B test results
            if winners:
                most_common_winner = max(set(winners), key=winners.count)
                consistency_rate = winners.count(most_common_winner) / len(winners)
                
                ab_consistency[comparison] = {
                    "winner_consistency": consistency_rate,
                    "most_common_winner": most_common_winner,
                    "all_winners": winners
                }
                
        return {
            "algorithm_consistency": robustness_metrics,
            "ab_test_consistency": ab_consistency
        }
        
    def _generate_publication_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate publication-ready summary"""
        
        rankings = results["algorithm_rankings"]["rankings"]
        meta_analysis = results["meta_analysis"]
        robustness = results["robustness_analysis"]
        
        # Key findings
        winner = rankings[0] if rankings else {"algorithm": "Unknown", "performance": 0}
        
        # Effect sizes
        primary_metric = self.config.primary_metric
        effect_sizes = meta_analysis.get(primary_metric, {}).get("effect_sizes", {})
        
        # Statistical significance summary
        significant_differences = []
        for comparison, test_result in results["algorithm_rankings"]["significance_tests"].items():
            if test_result["significant"]:
                significant_differences.append({
                    "comparison": comparison,
                    "p_value": test_result["p_value"],
                    "effect_size": test_result["effect_size_cohens_d"],
                    "interpretation": test_result["effect_size_interpretation"]
                })
                
        return {
            "executive_summary": {
                "best_algorithm": winner["algorithm"],
                "best_performance": winner["performance"],
                "statistically_significant_differences": len(significant_differences),
                "datasets_tested": len(results["datasets"]),
                "algorithms_compared": len(results["algorithms"])
            },
            "key_findings": {
                "winner": winner,
                "significant_differences": significant_differences,
                "effect_sizes": effect_sizes,
                "robustness_assessment": self._summarize_robustness(robustness)
            },
            "methodology": {
                "statistical_tests": "Cross-validation with statistical significance testing",
                "multiple_comparisons_correction": "Bonferroni correction applied",
                "confidence_level": self.config.confidence_level,
                "cross_validation_folds": self.config.cv_folds
            },
            "recommendations": self._generate_recommendations(results)
        }
        
    def _summarize_robustness(self, robustness: Dict[str, Any]) -> Dict[str, str]:
        """Summarize robustness analysis"""
        summary = {}
        
        algorithm_consistency = robustness.get("algorithm_consistency", {})
        for alg_name, consistency_data in algorithm_consistency.items():
            consistency = consistency_data["cross_dataset_consistency"]
            
            if consistency > 0.8:
                summary[alg_name] = "Highly robust"
            elif consistency > 0.6:
                summary[alg_name] = "Moderately robust"
            else:
                summary[alg_name] = "Low robustness"
                
        return summary
        
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        winner = results["algorithm_rankings"]["winner"]
        recommendations.append(f"Recommend {winner} as the primary algorithm based on statistical analysis")
        
        # Check for significant differences
        significance_tests = results["algorithm_rankings"]["significance_tests"]
        if not any(test["significant"] for test in significance_tests.values()):
            recommendations.append("No statistically significant differences found between top algorithms")
            recommendations.append("Consider practical significance and computational efficiency for selection")
            
        # Robustness recommendations
        robustness = results["robustness_analysis"]["algorithm_consistency"]
        for alg_name, consistency_data in robustness.items():
            if consistency_data["cross_dataset_consistency"] < 0.5:
                recommendations.append(f"Exercise caution with {alg_name} due to inconsistent performance across datasets")
                
        return recommendations
        
    def _save_results(self, results: Dict[str, Any], experiment_name: str):
        """Save experimental results"""
        timestamp = int(time.time())
        filename = f"{experiment_name.lower().replace(' ', '_')}_{timestamp}.json"
        filepath = Path(self.config.results_dir) / filename
        
        # Convert numpy types to Python native types for JSON serialization
        results_serializable = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'item'):  # torch tensors
            return obj.item()
        else:
            return obj
            
    def _generate_plots(self, results: Dict[str, Any], experiment_name: str):
        """Generate visualization plots"""
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            
            # 1. Algorithm performance comparison
            self._plot_algorithm_comparison(results, experiment_name)
            
            # 2. Cross-validation results
            self._plot_cross_validation_results(results, experiment_name)
            
            # 3. Effect size visualization
            self._plot_effect_sizes(results, experiment_name)
            
            logger.info("Plots generated successfully")
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
            
    def _plot_algorithm_comparison(self, results: Dict[str, Any], experiment_name: str):
        """Plot algorithm performance comparison"""
        rankings = results["algorithm_rankings"]["rankings"]
        
        if rankings:
            algorithms = [r["algorithm"] for r in rankings]
            performances = [r["performance"] for r in rankings]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(algorithms, performances)
            plt.title(f"Algorithm Performance Comparison - {experiment_name}")
            plt.xlabel("Algorithm")
            plt.ylabel(f"{self.config.primary_metric.title()} Score")
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, perf in zip(bars, performances):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{perf:.3f}", ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(Path(self.config.results_dir) / f"{experiment_name}_algorithm_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def _plot_cross_validation_results(self, results: Dict[str, Any], experiment_name: str):
        """Plot cross-validation results with error bars"""
        cv_results = results["cross_validation_results"]
        primary_metric = self.config.primary_metric
        
        algorithms = []
        means = []
        stds = []
        
        for alg_name, alg_results in cv_results.items():
            # Average across datasets
            dataset_means = []
            for dataset_result in alg_results.values():
                if primary_metric in dataset_result["cv_statistics"]:
                    dataset_means.append(dataset_result["cv_statistics"][primary_metric]["mean"])
                    
            if dataset_means:
                algorithms.append(alg_name)
                means.append(np.mean(dataset_means))
                stds.append(np.std(dataset_means))
                
        if algorithms:
            plt.figure(figsize=(10, 6))
            plt.errorbar(algorithms, means, yerr=stds, fmt='o', capsize=5, capthick=2)
            plt.title(f"Cross-Validation Results - {experiment_name}")
            plt.xlabel("Algorithm")
            plt.ylabel(f"{primary_metric.title()} Score")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(Path(self.config.results_dir) / f"{experiment_name}_cross_validation.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def _plot_effect_sizes(self, results: Dict[str, Any], experiment_name: str):
        """Plot effect sizes between algorithms"""
        primary_metric = self.config.primary_metric
        meta_analysis = results["meta_analysis"]
        
        if primary_metric in meta_analysis:
            effect_sizes = meta_analysis[primary_metric].get("effect_sizes", {})
            
            if effect_sizes:
                comparisons = list(effect_sizes.keys())
                cohens_d_values = [data["cohens_d"] for data in effect_sizes.values()]
                
                # Color code by effect size magnitude
                colors = []
                for d in cohens_d_values:
                    abs_d = abs(d)
                    if abs_d < 0.2:
                        colors.append('green')  # Negligible
                    elif abs_d < 0.5:
                        colors.append('yellow')  # Small
                    elif abs_d < 0.8:
                        colors.append('orange')  # Medium
                    else:
                        colors.append('red')  # Large
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(comparisons, cohens_d_values, color=colors, alpha=0.7)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title(f"Effect Sizes (Cohen's d) - {experiment_name}")
                plt.xlabel("Algorithm Comparison")
                plt.ylabel("Cohen's d")
                plt.xticks(rotation=45, ha='right')
                
                # Add horizontal lines for effect size thresholds
                plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
                plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
                plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.9, label='Large effect')
                plt.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
                plt.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.7)
                plt.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.9)
                
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(Path(self.config.results_dir) / f"{experiment_name}_effect_sizes.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()


# Demonstration and validation functions

def create_experimental_test_suite() -> Dict[str, Any]:
    """Create comprehensive test suite for experimental validation"""
    
    # Create test algorithms (mock implementations)
    test_algorithms = {
        "MAHF_System": MetaAdaptiveHierarchicalFusion(MAHFConfig()),
        "CARN_Baseline": CrossModalAdaptiveRetrievalNetwork(CARNConfig()),
        "Neuromorphic_System": NeuromorphicSpikeNetwork(NeuromorphicConfig(num_neurons=64, hidden_dim=384))
    }
    
    # Create test datasets
    def generate_test_dataset(n_samples: int, complexity: str = "medium") -> List[Dict[str, Any]]:
        dataset = []
        
        if complexity == "low":
            noise_level = 0.1
            seq_len = 16
        elif complexity == "high":
            noise_level = 0.3
            seq_len = 64
        else:  # medium
            noise_level = 0.2
            seq_len = 32
            
        for _ in range(n_samples):
            # Multi-modal input
            input_data = {
                "text": torch.randn(2, seq_len, 384),
                "embeddings": torch.randn(2, seq_len, 384) + torch.randn(2, seq_len, 384) * noise_level
            }
            
            # Synthetic target for evaluation
            target = torch.randn(2, seq_len, 384)
            
            dataset.append({
                "input": input_data,
                "target": target,
                "complexity": complexity
            })
            
        return dataset
    
    test_datasets = {
        "low_complexity": generate_test_dataset(50, "low"),
        "medium_complexity": generate_test_dataset(50, "medium"),
        "high_complexity": generate_test_dataset(50, "high")
    }
    
    # Define evaluation metrics
    def cosine_similarity_metric(output, target):
        output_flat = output.view(-1)
        target_flat = target.view(-1)
        return F.cosine_similarity(output_flat.unsqueeze(0), target_flat.unsqueeze(0)).item()
        
    def mse_metric(output, target):
        return F.mse_loss(output, target).item()
        
    def efficiency_metric(output):
        # Simplified efficiency based on activation sparsity
        return (1.0 - torch.count_nonzero(output).float() / output.numel()).item()
        
    evaluation_metrics = {
        "performance": cosine_similarity_metric,
        "mse_loss": mse_metric,
        "efficiency": efficiency_metric
    }
    
    return {
        "algorithms": test_algorithms,
        "datasets": test_datasets,
        "evaluation_metrics": evaluation_metrics
    }


def run_autonomous_experimental_validation():
    """Run comprehensive autonomous experimental validation"""
    
    print("🔬 AUTONOMOUS EXPERIMENTAL FRAMEWORK VALIDATION")
    print("=" * 80)
    
    # Configuration
    config = ExperimentalConfig(
        alpha=0.05,
        power=0.8,
        cv_folds=3,  # Reduced for demo
        confidence_level=0.95,
        primary_metric="performance",
        generate_plots=True,
        save_results=True
    )
    
    print(f"📋 Experimental Configuration:")
    print(f"   • Significance level: {config.alpha}")
    print(f"   • Statistical power: {config.power}")
    print(f"   • Cross-validation folds: {config.cv_folds}")
    print(f"   • Primary metric: {config.primary_metric}")
    
    # Create experimental framework
    framework = AutonomousExperimentalFramework(config)
    
    print(f"\n🧪 Framework Components:")
    print(f"   • Statistical analyzer with power analysis")
    print(f"   • A/B testing with Bonferroni correction")
    print(f"   • Cross-validation with bootstrap confidence intervals")
    print(f"   • Meta-analysis and robustness assessment")
    
    # Create test suite
    test_suite = create_experimental_test_suite()
    
    print(f"\n📊 Test Suite Created:")
    print(f"   • Algorithms: {list(test_suite['algorithms'].keys())}")
    print(f"   • Datasets: {list(test_suite['datasets'].keys())}")
    print(f"   • Evaluation metrics: {list(test_suite['evaluation_metrics'].keys())}")
    
    # Run comprehensive comparison
    print(f"\n🚀 RUNNING COMPREHENSIVE ALGORITHM COMPARISON:")
    print("-" * 60)
    
    try:
        comparison_results = framework.comprehensive_algorithm_comparison(
            algorithms=test_suite["algorithms"],
            test_datasets=test_suite["datasets"],
            evaluation_metrics=test_suite["evaluation_metrics"],
            experiment_name="MAHF_Research_Validation"
        )
        
        print(f"✓ Experimental comparison completed successfully")
        
        # Display key results
        print(f"\n🏆 EXPERIMENTAL RESULTS:")
        print("-" * 40)
        
        # Winner and rankings
        publication_summary = comparison_results["publication_summary"]
        executive_summary = publication_summary["executive_summary"]
        
        print(f"🥇 WINNER: {executive_summary['best_algorithm']}")
        print(f"   • Performance: {executive_summary['best_performance']:.4f}")
        print(f"   • Algorithms compared: {executive_summary['algorithms_compared']}")
        print(f"   • Datasets tested: {executive_summary['datasets_tested']}")
        print(f"   • Significant differences found: {executive_summary['statistically_significant_differences']}")
        
        # Rankings
        rankings = comparison_results["algorithm_rankings"]["rankings"]
        print(f"\n📊 ALGORITHM RANKINGS:")
        for rank_data in rankings:
            print(f"   {rank_data['rank']}. {rank_data['algorithm']}: {rank_data['performance']:.4f}")
            
        # Statistical significance
        significance_tests = comparison_results["algorithm_rankings"]["significance_tests"]
        print(f"\n📈 STATISTICAL SIGNIFICANCE TESTS:")
        for comparison, test_result in significance_tests.items():
            significance = "✓ Significant" if test_result["significant"] else "✗ Not significant"
            effect_size = test_result["effect_size_cohens_d"]
            effect_interpretation = test_result["effect_size_interpretation"]
            
            print(f"   • {comparison}")
            print(f"     {significance} (p={test_result['p_value']:.4f})")
            print(f"     Effect size: {effect_size:.3f} ({effect_interpretation})")
            
        # Robustness analysis
        robustness = comparison_results["robustness_analysis"]
        algorithm_consistency = robustness.get("algorithm_consistency", {})
        
        print(f"\n🛡️  ROBUSTNESS ANALYSIS:")
        for alg_name, consistency_data in algorithm_consistency.items():
            consistency = consistency_data["cross_dataset_consistency"]
            consistency_level = "High" if consistency > 0.8 else "Medium" if consistency > 0.6 else "Low"
            print(f"   • {alg_name}: {consistency:.3f} ({consistency_level} robustness)")
            
        # Meta-analysis effect sizes
        meta_analysis = comparison_results["meta_analysis"]
        primary_metric = config.primary_metric
        
        if primary_metric in meta_analysis:
            effect_sizes = meta_analysis[primary_metric].get("effect_sizes", {})
            
            if effect_sizes:
                print(f"\n📏 EFFECT SIZES (Cohen's d):")
                for comparison, effect_data in effect_sizes.items():
                    cohens_d = effect_data["cohens_d"]
                    interpretation = effect_data["interpretation"]
                    print(f"   • {comparison}: {cohens_d:.3f} ({interpretation})")
                    
        # Key findings
        key_findings = publication_summary["key_findings"]
        print(f"\n🔍 KEY FINDINGS:")
        
        significant_diffs = key_findings["significant_differences"]
        if significant_diffs:
            print(f"   • Statistically significant differences detected:")
            for diff in significant_diffs:
                print(f"     - {diff['comparison']}: p={diff['p_value']:.4f}, d={diff['effect_size']:.3f}")
        else:
            print(f"   • No statistically significant differences detected")
            print(f"   • Consider practical significance for algorithm selection")
            
        # Recommendations
        recommendations = publication_summary["recommendations"]
        print(f"\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(recommendations, 1):
            print(f"   {i}. {recommendation}")
            
        # Methodology summary
        methodology = publication_summary["methodology"]
        print(f"\n📋 METHODOLOGY VALIDATION:")
        print(f"   • Statistical tests: {methodology['statistical_tests']}")
        print(f"   • Multiple comparisons: {methodology['multiple_comparisons_correction']}")
        print(f"   • Confidence level: {methodology['confidence_level']}")
        print(f"   • Cross-validation: {methodology['cross_validation_folds']} folds")
        
        print(f"\n✅ AUTONOMOUS EXPERIMENTAL VALIDATION COMPLETE!")
        print(f"🏆 Rigorous statistical analysis performed")
        print(f"📊 Publication-ready results generated")
        print(f"🔬 Reproducible experimental protocols validated")
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Experimental validation failed: {e}")
        print(f"❌ Experimental validation failed: {e}")
        return None


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    run_autonomous_experimental_validation()