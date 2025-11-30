"""
Offline Experiment Module for LSED
Runs experiments on the offline scenario (message block M0).

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

Offline scenario:
- First 7 days of data collected as M0
- Train and evaluate on M0
- Report NMI and AMI with mean ± std over multiple runs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time
import json
from datetime import datetime

# Import LSED modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, MessageBlock
from models.lsed import LSED
from utils.metrics import ClusteringMetrics, ExperimentResults

logger = logging.getLogger(__name__)


class OfflineExperiment:
    """
    Offline experiment runner for LSED.
    
    Evaluates LSED on the offline scenario (M0 message block).
    
    Procedure:
    1. Load and preprocess data
    2. Run multiple experiments with different random seeds
    3. Compute mean and std of NMI and AMI
    4. Compare with baseline results
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_path: str,
        results_dir: str = "results/offline",
        use_mock_llm: bool = False
    ):
        """
        Initialize offline experiment.
        
        Args:
            config: Configuration dictionary
            data_path: Path to input CSV file
            results_dir: Directory to save results
            use_mock_llm: Whether to use mock LLM
        """
        self.config = config
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock_llm = use_mock_llm
        
        # Experiment settings
        self.num_runs = config.get('evaluation', {}).get('num_runs', 5)
        
        # Data loader
        self.data_loader = DataLoader(config)
        
        # Results storage
        self.results = ExperimentResults("Offline Experiment")
    
    def load_data(self) -> MessageBlock:
        """
        Load data and return the offline block (M0).
        
        Returns:
            MessageBlock for M0
        """
        logger.info(f"Loading data from {self.data_path}")
        
        self.data_loader.load_data(self.data_path)
        offline_block, _ = self.data_loader.split_into_blocks()
        
        self.data_loader.print_statistics()
        
        return offline_block
    
    def prepare_data(
        self,
        block: MessageBlock
    ) -> Tuple[List[str], List, np.ndarray]:
        """
        Prepare data for LSED.
        
        Args:
            block: Message block
            
        Returns:
            Tuple of (texts, timestamps, labels)
        """
        data = block.data
        
        texts = data[self.data_loader.text_column].tolist()
        timestamps = data[self.data_loader.timestamp_column].tolist()
        labels = data[self.data_loader.event_column].values
        
        # Convert labels to contiguous integers
        unique_labels = np.unique(labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        
        return texts, timestamps, labels
    
    def run_single_experiment(
        self,
        texts: List[str],
        timestamps: List,
        labels: np.ndarray,
        run_id: int
    ) -> Dict[str, float]:
        """
        Run a single experiment.
        
        Args:
            texts: Message texts
            timestamps: Timestamps
            labels: Ground truth labels
            run_id: Run identifier
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Running experiment {run_id + 1}/{self.num_runs}")
        
        # Set seed for this run
        seed = self.config.get('training', {}).get('seed', 42) + run_id
        np.random.seed(seed)
        
        # Modify config with new seed
        run_config = self.config.copy()
        run_config['training'] = run_config.get('training', {}).copy()
        run_config['training']['seed'] = seed
        
        # Initialize LSED
        lsed = LSED(run_config, use_mock_llm=self.use_mock_llm)
        
        # Get number of clusters
        n_clusters = len(np.unique(labels))
        
        # Step 1: Summarize
        start_time = time.time()
        summarized = lsed.summarize_messages(texts, show_progress=True)
        summarize_time = time.time() - start_time
        
        # Step 2: Vectorize
        start_time = time.time()
        vectors = lsed.vectorize_messages(summarized, timestamps)
        vectorize_time = time.time() - start_time
        
        # Step 3: Split data
        n = len(vectors)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.9)
        
        train_vectors = vectors[:train_idx]
        train_labels = labels[:train_idx]
        val_vectors = vectors[train_idx:val_idx]
        val_labels = labels[train_idx:val_idx]
        test_vectors = vectors[val_idx:]
        test_labels = labels[val_idx:]
        
        # Step 4: Train
        start_time = time.time()
        history = lsed.train_model(
            train_vectors, train_labels,
            val_vectors, val_labels,
            verbose=True
        )
        train_time = time.time() - start_time
        
        # Step 5: Evaluate on test set
        start_time = time.time()
        embeddings = lsed.encode_hyperbolic(test_vectors, training=False)
        pred_labels = lsed.cluster_events(embeddings, n_clusters=n_clusters)
        eval_time = time.time() - start_time
        
        # Compute metrics
        metrics = ClusteringMetrics.compute_primary(test_labels, pred_labels)
        
        # Add timing info
        metrics['summarize_time'] = summarize_time
        metrics['vectorize_time'] = vectorize_time
        metrics['train_time'] = train_time
        metrics['eval_time'] = eval_time
        metrics['total_time'] = summarize_time + vectorize_time + train_time + eval_time
        
        logger.info(f"Run {run_id + 1}: NMI={metrics['nmi']:.4f}, AMI={metrics['ami']:.4f}")
        
        return metrics
    
    def run(self) -> ExperimentResults:
        """
        Run the full offline experiment.
        
        Returns:
            ExperimentResults with all runs
        """
        logger.info("Starting Offline Experiment")
        logger.info("="*60)
        
        # Load data
        offline_block = self.load_data()
        texts, timestamps, labels = self.prepare_data(offline_block)
        
        logger.info(f"Data prepared: {len(texts)} messages, {len(np.unique(labels))} events")
        
        # Run multiple experiments
        all_metrics = []
        
        for run_id in range(self.num_runs):
            metrics = self.run_single_experiment(texts, timestamps, labels, run_id)
            self.results.add_run(metrics, block_name="M0")
            all_metrics.append(metrics)
        
        # Print summary
        self.results.print_summary(['nmi', 'ami'])
        
        # Save results
        self._save_results(all_metrics)
        
        return self.results
    
    def _save_results(self, metrics: List[Dict[str, float]]):
        """Save results to file."""
        # Summary
        summary = {
            'experiment': 'offline',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'llm_model': self.config.get('llm', {}).get('model', 'llama3.1'),
                'prompt_type': self.config.get('prompts', {}).get('prompt_type', 'summarize'),
                'vectorizer': self.config.get('vectorization', {}).get('model', 'sbert'),
                'hyperbolic_model': self.config.get('hyperbolic', {}).get('model', 'poincare'),
                'num_runs': self.num_runs
            },
            'results': {
                'nmi': {
                    'mean': float(np.mean([m['nmi'] for m in metrics])),
                    'std': float(np.std([m['nmi'] for m in metrics]))
                },
                'ami': {
                    'mean': float(np.mean([m['ami'] for m in metrics])),
                    'std': float(np.std([m['ami'] for m in metrics]))
                }
            },
            'runs': metrics
        }
        
        # Save JSON
        output_path = self.results_dir / 'offline_results.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print paper-style results table
        print("\n" + "="*60)
        print("OFFLINE EXPERIMENT RESULTS (Paper Style)")
        print("="*60)
        
        nmi_mean, nmi_std = self.results.get_mean_std('nmi')
        ami_mean, ami_std = self.results.get_mean_std('ami')
        
        print(f"\n{'Method':<20} {'NMI':>15} {'AMI':>15}")
        print("-"*50)
        print(f"{'LSED':<20} {nmi_mean:.2f}±{nmi_std:.2f}{' ':>5} {ami_mean:.2f}±{ami_std:.2f}")
        print("="*60)


def run_offline_experiment(
    config_path: str,
    data_path: str,
    use_mock_llm: bool = False
):
    """
    Convenience function to run offline experiment.
    
    Args:
        config_path: Path to config file
        data_path: Path to data CSV
        use_mock_llm: Whether to use mock LLM
    """
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run experiment
    experiment = OfflineExperiment(
        config=config,
        data_path=data_path,
        use_mock_llm=use_mock_llm
    )
    
    results = experiment.run()
    
    return results


if __name__ == "__main__":
    import yaml
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run LSED Offline Experiment")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default="data/sample_data.csv")
    parser.add_argument("--mock-llm", action="store_true")
    args = parser.parse_args()
    
    # Create sample data if needed
    from data.data_loader import create_sample_data
    if not Path(args.data).exists():
        create_sample_data(args.data, n_messages=500, n_events=20)
    
    # Run experiment
    results = run_offline_experiment(
        config_path=args.config,
        data_path=args.data,
        use_mock_llm=args.mock_llm
    )
