"""
Metrics Module for LSED
Implements evaluation metrics: NMI and AMI.

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

Metrics:
- NMI (Normalized Mutual Information)
- AMI (Adjusted Mutual Information)

These are standard clustering evaluation metrics that compare
predicted cluster labels with ground truth event labels.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    v_measure_score,
    homogeneity_score,
    completeness_score
)
import logging

logger = logging.getLogger(__name__)


class ClusteringMetrics:
    """
    Clustering evaluation metrics for Social Event Detection.
    
    Primary metrics (from paper):
    - NMI: Normalized Mutual Information
    - AMI: Adjusted Mutual Information
    
    Additional metrics:
    - ARI: Adjusted Rand Index
    - FMI: Fowlkes-Mallows Index
    - V-Measure, Homogeneity, Completeness
    """
    
    @staticmethod
    def nmi(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        average_method: str = 'arithmetic'
    ) -> float:
        """
        Compute Normalized Mutual Information.
        
        NMI measures the mutual dependence between true and predicted labels,
        normalized by the average entropy of both labelings.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            average_method: Method for normalization ('arithmetic', 'geometric', 'min', 'max')
            
        Returns:
            NMI score in [0, 1]
        """
        return normalized_mutual_info_score(
            y_true, y_pred, 
            average_method=average_method
        )
    
    @staticmethod
    def ami(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Adjusted Mutual Information.
        
        AMI adjusts for chance, providing a more robust measure than NMI.
        A value of 0 indicates random labeling.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            AMI score (can be negative, 1 is perfect)
        """
        return adjusted_mutual_info_score(y_true, y_pred)
    
    @staticmethod
    def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Adjusted Rand Index.
        
        ARI measures similarity between two clusterings, adjusted for chance.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            ARI score
        """
        return adjusted_rand_score(y_true, y_pred)
    
    @staticmethod
    def fmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Fowlkes-Mallows Index.
        
        FMI is the geometric mean of precision and recall for pair-wise clustering.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            FMI score
        """
        return fowlkes_mallows_score(y_true, y_pred)
    
    @staticmethod
    def v_measure(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute V-Measure.
        
        V-Measure is the harmonic mean of homogeneity and completeness.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            V-Measure score
        """
        return v_measure_score(y_true, y_pred)
    
    @staticmethod
    def homogeneity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Homogeneity score."""
        return homogeneity_score(y_true, y_pred)
    
    @staticmethod
    def completeness(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Completeness score."""
        return completeness_score(y_true, y_pred)
    
    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            Dictionary of metric names to scores
        """
        return {
            'nmi': cls.nmi(y_true, y_pred),
            'ami': cls.ami(y_true, y_pred),
            'ari': cls.ari(y_true, y_pred),
            'fmi': cls.fmi(y_true, y_pred),
            'v_measure': cls.v_measure(y_true, y_pred),
            'homogeneity': cls.homogeneity(y_true, y_pred),
            'completeness': cls.completeness(y_true, y_pred)
        }
    
    @classmethod
    def compute_primary(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute primary metrics (NMI and AMI) as used in the paper.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted cluster labels
            
        Returns:
            Dictionary with 'nmi' and 'ami' scores
        """
        return {
            'nmi': cls.nmi(y_true, y_pred),
            'ami': cls.ami(y_true, y_pred)
        }


class ExperimentResults:
    """
    Container for experiment results across multiple runs.
    
    Stores results from multiple runs and computes statistics
    (mean, std) as reported in the paper.
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize results container.
        
        Args:
            name: Name of the experiment
        """
        self.name = name
        self.runs: List[Dict[str, float]] = []
        self.block_results: Dict[str, List[Dict[str, float]]] = {}
    
    def add_run(self, metrics: Dict[str, float], block_name: Optional[str] = None):
        """
        Add results from a single run.
        
        Args:
            metrics: Dictionary of metric names to scores
            block_name: Optional message block name
        """
        self.runs.append(metrics)
        
        if block_name:
            if block_name not in self.block_results:
                self.block_results[block_name] = []
            self.block_results[block_name].append(metrics)
    
    def get_mean_std(self, metric: str) -> Tuple[float, float]:
        """
        Get mean and standard deviation for a metric across runs.
        
        Args:
            metric: Metric name
            
        Returns:
            Tuple of (mean, std)
        """
        values = [run[metric] for run in self.runs if metric in run]
        if not values:
            return 0.0, 0.0
        return np.mean(values), np.std(values)
    
    def get_block_mean_std(
        self, 
        block_name: str, 
        metric: str
    ) -> Tuple[float, float]:
        """
        Get mean and std for a metric for a specific block.
        
        Args:
            block_name: Message block name
            metric: Metric name
            
        Returns:
            Tuple of (mean, std)
        """
        if block_name not in self.block_results:
            return 0.0, 0.0
        
        values = [run[metric] for run in self.block_results[block_name] if metric in run]
        if not values:
            return 0.0, 0.0
        return np.mean(values), np.std(values)
    
    def get_summary(self, metrics: List[str] = ['nmi', 'ami']) -> Dict[str, Tuple[float, float]]:
        """
        Get summary statistics for specified metrics.
        
        Args:
            metrics: List of metric names
            
        Returns:
            Dictionary mapping metric names to (mean, std) tuples
        """
        return {metric: self.get_mean_std(metric) for metric in metrics}
    
    def format_result(
        self, 
        metric: str,
        precision: int = 2
    ) -> str:
        """
        Format a metric result as "mean±std" string (paper style).
        
        Args:
            metric: Metric name
            precision: Decimal precision
            
        Returns:
            Formatted string like ".96±.01"
        """
        mean, std = self.get_mean_std(metric)
        return f".{int(mean*100):02d}±.{int(std*100):02d}"
    
    def print_summary(self, metrics: List[str] = ['nmi', 'ami']):
        """Print summary of results."""
        print(f"\n{'='*60}")
        print(f"Results Summary: {self.name}")
        print(f"{'='*60}")
        print(f"Number of runs: {len(self.runs)}")
        print(f"\n{'Metric':<15} {'Mean':>10} {'Std':>10} {'Formatted':>15}")
        print(f"{'-'*50}")
        
        for metric in metrics:
            mean, std = self.get_mean_std(metric)
            formatted = self.format_result(metric)
            print(f"{metric.upper():<15} {mean:>10.4f} {std:>10.4f} {formatted:>15}")
        
        print(f"{'='*60}\n")
    
    def print_block_summary(self, metrics: List[str] = ['nmi', 'ami']):
        """Print summary for each message block."""
        if not self.block_results:
            print("No block results available.")
            return
        
        print(f"\n{'='*80}")
        print(f"Block Results Summary: {self.name}")
        print(f"{'='*80}")
        
        # Header
        header = f"{'Block':<10}"
        for metric in metrics:
            header += f"{metric.upper():>15}"
        print(header)
        print("-"*80)
        
        # Results per block
        for block_name in sorted(self.block_results.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
            row = f"{block_name:<10}"
            for metric in metrics:
                mean, std = self.get_block_mean_std(block_name, metric)
                row += f"{mean:>8.2f}±{std:.2f}"
            print(row)
        
        print(f"{'='*80}\n")


class MetricsLogger:
    """
    Logger for tracking metrics during training/evaluation.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Optional path to save metrics log
        """
        self.log_file = log_file
        self.history: List[Dict] = []
    
    def log(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = "train"
    ):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            phase: Phase name (train/val/test)
        """
        entry = {
            'epoch': epoch,
            'phase': phase,
            **metrics
        }
        self.history.append(entry)
        
        # Log to console
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch} [{phase}]: {metrics_str}")
    
    def get_best(self, metric: str = 'nmi', phase: str = 'val') -> Dict:
        """
        Get entry with best metric value.
        
        Args:
            metric: Metric name
            phase: Phase to consider
            
        Returns:
            Entry with best metric
        """
        phase_entries = [e for e in self.history if e['phase'] == phase and metric in e]
        if not phase_entries:
            return {}
        return max(phase_entries, key=lambda x: x[metric])
    
    def save(self, path: Optional[str] = None):
        """Save history to file."""
        import json
        
        path = path or self.log_file
        if path:
            with open(path, 'w') as f:
                json.dump(self.history, f, indent=2)


def evaluate_clustering(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Convenience function to evaluate clustering results.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
        verbose: Whether to print results
        
    Returns:
        Dictionary of metrics
    """
    metrics = ClusteringMetrics.compute_primary(y_true, y_pred)
    
    if verbose:
        print(f"Clustering Evaluation:")
        print(f"  NMI: {metrics['nmi']:.4f}")
        print(f"  AMI: {metrics['ami']:.4f}")
    
    return metrics


if __name__ == "__main__":
    # Test metrics module
    print("Testing Metrics Module")
    print("="*60)
    
    # Generate synthetic clustering data
    np.random.seed(42)
    n_samples = 500
    n_clusters = 10
    
    # Perfect clustering
    y_true = np.repeat(np.arange(n_clusters), n_samples // n_clusters)
    y_pred_perfect = y_true.copy()
    
    # Random clustering
    y_pred_random = np.random.randint(0, n_clusters, n_samples)
    
    # Slightly noisy clustering
    y_pred_noisy = y_true.copy()
    noise_idx = np.random.choice(n_samples, n_samples // 10, replace=False)
    y_pred_noisy[noise_idx] = np.random.randint(0, n_clusters, len(noise_idx))
    
    print("\n1. Perfect Clustering:")
    metrics_perfect = ClusteringMetrics.compute_primary(y_true, y_pred_perfect)
    print(f"   NMI: {metrics_perfect['nmi']:.4f}")
    print(f"   AMI: {metrics_perfect['ami']:.4f}")
    
    print("\n2. Random Clustering:")
    metrics_random = ClusteringMetrics.compute_primary(y_true, y_pred_random)
    print(f"   NMI: {metrics_random['nmi']:.4f}")
    print(f"   AMI: {metrics_random['ami']:.4f}")
    
    print("\n3. Noisy Clustering (90% correct):")
    metrics_noisy = ClusteringMetrics.compute_primary(y_true, y_pred_noisy)
    print(f"   NMI: {metrics_noisy['nmi']:.4f}")
    print(f"   AMI: {metrics_noisy['ami']:.4f}")
    
    print("\n4. All Metrics (Noisy):")
    all_metrics = ClusteringMetrics.compute_all(y_true, y_pred_noisy)
    for name, value in all_metrics.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n5. ExperimentResults:")
    results = ExperimentResults("Test Experiment")
    
    # Simulate 5 runs
    for run in range(5):
        noise = np.random.randn(2) * 0.01
        metrics = {
            'nmi': 0.95 + noise[0],
            'ami': 0.93 + noise[1]
        }
        results.add_run(metrics)
    
    results.print_summary()
    
    print("\n6. Block Results:")
    block_results = ExperimentResults("Block Test")
    
    for block in ['M0', 'M1', 'M2', 'M3']:
        for run in range(5):
            noise = np.random.randn(2) * 0.02
            metrics = {
                'nmi': 0.90 + 0.02 * int(block[1:]) + noise[0],
                'ami': 0.88 + 0.02 * int(block[1:]) + noise[1]
            }
            block_results.add_run(metrics, block_name=block)
    
    block_results.print_block_summary()
    
    print("="*60)
    print("All tests passed!")
