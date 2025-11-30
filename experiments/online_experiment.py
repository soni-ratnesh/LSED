"""
Online Experiment Module for LSED
Runs experiments on the online (incremental) scenario.

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

Online scenario:
- First 7 days = M0 (offline training)
- Remaining days = M1, M2, ..., Mn (online processing)
- Model updates according to window size w
- Report NMI and AMI for each block with mean ± std
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time
import json
from datetime import datetime
from tqdm import tqdm

# Import LSED modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, MessageBlock
from models.lsed import LSED, IncrementalLSED
from utils.metrics import ClusteringMetrics, ExperimentResults

logger = logging.getLogger(__name__)


class OnlineExperiment:
    """
    Online experiment runner for LSED.
    
    Implements incremental social event detection:
    1. Train on M0 (offline)
    2. Process M1, M2, ..., Mn incrementally
    3. Update model according to window size
    4. Report metrics for each block
    
    As per Algorithm 1 in the paper:
    - If i % w == 0: Update parameters (train)
    - Else: Just run LSED iteration (clustering)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_path: str,
        results_dir: str = "results/online",
        use_mock_llm: bool = False
    ):
        """
        Initialize online experiment.
        
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
        self.window_size = config.get('incremental', {}).get('window_size', 1)
        
        # Data loader
        self.data_loader = DataLoader(config)
        
        # Results storage
        self.results = ExperimentResults("Online Experiment")
        self.block_results: Dict[str, ExperimentResults] = {}
    
    def load_data(self) -> Tuple[MessageBlock, List[MessageBlock]]:
        """
        Load data and return offline and online blocks.
        
        Returns:
            Tuple of (offline_block, list of online_blocks)
        """
        logger.info(f"Loading data from {self.data_path}")
        
        self.data_loader.load_data(self.data_path)
        offline_block, online_blocks = self.data_loader.split_into_blocks()
        
        self.data_loader.print_statistics()
        
        return offline_block, online_blocks
    
    def prepare_block(
        self,
        block: MessageBlock
    ) -> Tuple[List[str], List, np.ndarray]:
        """
        Prepare data from a message block.
        
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
        offline_block: MessageBlock,
        online_blocks: List[MessageBlock],
        run_id: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Run a single online experiment.
        
        Args:
            offline_block: M0 block for initial training
            online_blocks: List of M1, M2, ... blocks
            run_id: Run identifier
            
        Returns:
            Dictionary mapping block names to metrics
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
        lsed = IncrementalLSED(run_config, use_mock_llm=self.use_mock_llm)
        
        block_metrics = {}
        
        # Step 1: Train on M0 (offline)
        logger.info("Training on M0 (offline)...")
        texts, timestamps, labels = self.prepare_block(offline_block)
        
        m0_metrics = lsed.process_block(
            "M0",
            texts,
            timestamps,
            labels,
            summarize=True,
            train=True
        )
        block_metrics["M0"] = m0_metrics
        
        # Store M0 results
        if "M0" not in self.block_results:
            self.block_results["M0"] = ExperimentResults("M0")
        self.block_results["M0"].add_run(m0_metrics)
        
        # Step 2: Process online blocks
        logger.info(f"Processing {len(online_blocks)} online blocks...")
        
        for block in tqdm(online_blocks, desc="Online blocks"):
            texts, timestamps, labels = self.prepare_block(block)
            
            # Determine if we should update
            should_train = lsed.should_update()
            
            metrics = lsed.process_block(
                block.block_name,
                texts,
                timestamps,
                labels,
                summarize=True,
                train=should_train
            )
            
            block_metrics[block.block_name] = metrics
            
            # Store block results
            if block.block_name not in self.block_results:
                self.block_results[block.block_name] = ExperimentResults(block.block_name)
            self.block_results[block.block_name].add_run(metrics)
        
        # Compute average metrics for this run
        avg_nmi = np.mean([m['nmi'] for m in block_metrics.values()])
        avg_ami = np.mean([m['ami'] for m in block_metrics.values()])
        
        logger.info(f"Run {run_id + 1}: Average NMI={avg_nmi:.4f}, AMI={avg_ami:.4f}")
        
        # Store run results
        self.results.add_run({'nmi': avg_nmi, 'ami': avg_ami})
        
        return block_metrics
    
    def run(self) -> Tuple[ExperimentResults, Dict[str, ExperimentResults]]:
        """
        Run the full online experiment.
        
        Returns:
            Tuple of (overall results, block-wise results)
        """
        logger.info("Starting Online Experiment")
        logger.info("="*60)
        logger.info(f"Window size: {self.window_size}")
        logger.info(f"Number of runs: {self.num_runs}")
        
        # Load data
        offline_block, online_blocks = self.load_data()
        
        logger.info(f"Offline block: {offline_block}")
        logger.info(f"Online blocks: {len(online_blocks)}")
        
        # Run multiple experiments
        all_block_metrics = []
        
        for run_id in range(self.num_runs):
            block_metrics = self.run_single_experiment(
                offline_block,
                online_blocks,
                run_id
            )
            all_block_metrics.append(block_metrics)
        
        # Print summary
        self.results.print_summary(['nmi', 'ami'])
        self._print_block_summary()
        
        # Save results
        self._save_results(all_block_metrics)
        
        return self.results, self.block_results
    
    def _print_block_summary(self):
        """Print summary for each block."""
        print("\n" + "="*80)
        print("ONLINE EXPERIMENT - BLOCK RESULTS")
        print("="*80)
        
        # Header
        print(f"\n{'Block':<10} {'#Events':>10} {'NMI':>15} {'AMI':>15}")
        print("-"*50)
        
        # Get all blocks in order
        blocks = sorted(
            self.block_results.keys(),
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
        )
        
        for block_name in blocks:
            block_result = self.block_results[block_name]
            nmi_mean, nmi_std = block_result.get_mean_std('nmi')
            ami_mean, ami_std = block_result.get_mean_std('ami')
            
            print(f"{block_name:<10} {'-':>10} {nmi_mean:.2f}±{nmi_std:.2f}{' ':>3} {ami_mean:.2f}±{ami_std:.2f}")
        
        print("="*80)
        
        # Average
        print("\nAverage across all blocks:")
        avg_nmi, std_nmi = self.results.get_mean_std('nmi')
        avg_ami, std_ami = self.results.get_mean_std('ami')
        print(f"  NMI: {avg_nmi:.2f} ± {std_nmi:.2f}")
        print(f"  AMI: {avg_ami:.2f} ± {std_ami:.2f}")
    
    def _save_results(self, all_block_metrics: List[Dict]):
        """Save results to file."""
        # Prepare block summaries
        block_summaries = {}
        for block_name, block_result in self.block_results.items():
            nmi_mean, nmi_std = block_result.get_mean_std('nmi')
            ami_mean, ami_std = block_result.get_mean_std('ami')
            
            block_summaries[block_name] = {
                'nmi': {'mean': float(nmi_mean), 'std': float(nmi_std)},
                'ami': {'mean': float(ami_mean), 'std': float(ami_std)}
            }
        
        # Overall summary
        avg_nmi, std_nmi = self.results.get_mean_std('nmi')
        avg_ami, std_ami = self.results.get_mean_std('ami')
        
        summary = {
            'experiment': 'online',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'llm_model': self.config.get('llm', {}).get('model', 'llama3.1'),
                'prompt_type': self.config.get('prompts', {}).get('prompt_type', 'summarize'),
                'vectorizer': self.config.get('vectorization', {}).get('model', 'sbert'),
                'hyperbolic_model': self.config.get('hyperbolic', {}).get('model', 'poincare'),
                'window_size': self.window_size,
                'num_runs': self.num_runs
            },
            'overall_results': {
                'nmi': {'mean': float(avg_nmi), 'std': float(std_nmi)},
                'ami': {'mean': float(avg_ami), 'std': float(std_ami)}
            },
            'block_results': block_summaries,
            'runs': all_block_metrics
        }
        
        # Save JSON
        output_path = self.results_dir / 'online_results.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


class WindowSizeAnalysis:
    """
    Analysis of different window sizes for online experiment.
    
    Tests window sizes: 1, 2, 3, 4, 5, ∞ (as shown in Figure 6 of paper)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_path: str,
        results_dir: str = "results/window_analysis",
        use_mock_llm: bool = False
    ):
        """Initialize window size analysis."""
        self.config = config
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock_llm = use_mock_llm
        
        # Window sizes to test (∞ is represented as large number)
        self.window_sizes = [1, 2, 3, 4, 5, 1000]  # 1000 = infinity
    
    def run(self) -> Dict[int, ExperimentResults]:
        """
        Run window size analysis.
        
        Returns:
            Dictionary mapping window size to results
        """
        logger.info("Starting Window Size Analysis")
        logger.info("="*60)
        
        results = {}
        
        for w in self.window_sizes:
            logger.info(f"\nTesting window size w = {w if w < 1000 else '∞'}")
            
            # Update config with window size
            config = self.config.copy()
            config['incremental'] = config.get('incremental', {}).copy()
            config['incremental']['window_size'] = w
            
            # Run online experiment
            experiment = OnlineExperiment(
                config=config,
                data_path=self.data_path,
                results_dir=str(self.results_dir / f"w_{w}"),
                use_mock_llm=self.use_mock_llm
            )
            
            exp_results, _ = experiment.run()
            results[w] = exp_results
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[int, ExperimentResults]):
        """Print comparison of different window sizes."""
        print("\n" + "="*60)
        print("WINDOW SIZE ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n{'Window Size':>15} {'NMI':>15} {'AMI':>15}")
        print("-"*45)
        
        for w in self.window_sizes:
            w_str = str(w) if w < 1000 else "∞"
            result = results[w]
            nmi_mean, nmi_std = result.get_mean_std('nmi')
            ami_mean, ami_std = result.get_mean_std('ami')
            
            print(f"{w_str:>15} {nmi_mean:.2f}±{nmi_std:.2f}{' ':>3} {ami_mean:.2f}±{ami_std:.2f}")
        
        print("="*60)


def run_online_experiment(
    config_path: str,
    data_path: str,
    use_mock_llm: bool = False
):
    """
    Convenience function to run online experiment.
    
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
    experiment = OnlineExperiment(
        config=config,
        data_path=data_path,
        use_mock_llm=use_mock_llm
    )
    
    results, block_results = experiment.run()
    
    return results, block_results


if __name__ == "__main__":
    import yaml
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run LSED Online Experiment")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default="data/sample_data.csv")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--window-analysis", action="store_true",
                       help="Run window size analysis")
    args = parser.parse_args()
    
    # Create sample data if needed
    from data.data_loader import create_sample_data
    if not Path(args.data).exists():
        create_sample_data(args.data, n_messages=1000, n_events=30)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.window_analysis:
        # Run window size analysis
        analysis = WindowSizeAnalysis(
            config=config,
            data_path=args.data,
            use_mock_llm=args.mock_llm
        )
        analysis.run()
    else:
        # Run standard online experiment
        results, block_results = run_online_experiment(
            config_path=args.config,
            data_path=args.data,
            use_mock_llm=args.mock_llm
        )
