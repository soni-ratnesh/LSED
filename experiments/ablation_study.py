"""
Ablation Study Module for LSED
Evaluates the impact of different components on LSED performance.

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

Ablation studies include:
1. Impact of Vectorization Methods (SBERT vs Word2Vec, with/without time)
2. Effect of Hierarchical Structure and Hyperbolic Embedding
3. Impact of LLM summarization
4. Comparison of prompt types (summarize vs paraphrase)
5. Comparison of hyperbolic models (Poincaré vs Hyperboloid)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time
import json
from datetime import datetime
from copy import deepcopy

# Import LSED modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, MessageBlock
from models.lsed import LSED
from utils.metrics import ClusteringMetrics, ExperimentResults

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Ablation study runner for LSED.
    
    Evaluates the contribution of each component to the overall performance.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_path: str,
        results_dir: str = "results/ablation",
        use_mock_llm: bool = False
    ):
        """
        Initialize ablation study.
        
        Args:
            config: Base configuration dictionary
            data_path: Path to input CSV file
            results_dir: Directory to save results
            use_mock_llm: Whether to use mock LLM
        """
        self.base_config = config
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock_llm = use_mock_llm
        
        # Data loader
        self.data_loader = DataLoader(config)
        
        # Results storage
        self.all_results: Dict[str, ExperimentResults] = {}
    
    def load_data(self) -> Tuple[List[str], List, np.ndarray]:
        """Load and prepare data for ablation study."""
        self.data_loader.load_data(self.data_path)
        offline_block, _ = self.data_loader.split_into_blocks()
        
        data = offline_block.data
        texts = data[self.data_loader.text_column].tolist()
        timestamps = data[self.data_loader.timestamp_column].tolist()
        labels = data[self.data_loader.event_column].values
        
        # Convert labels
        unique_labels = np.unique(labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        
        return texts, timestamps, labels
    
    def run_single_config(
        self,
        config: Dict[str, Any],
        texts: List[str],
        timestamps: List,
        labels: np.ndarray,
        name: str,
        num_runs: int = 5
    ) -> ExperimentResults:
        """
        Run experiment with a single configuration.
        
        Args:
            config: Configuration dictionary
            texts: Message texts
            timestamps: Timestamps
            labels: Ground truth labels
            name: Experiment name
            num_runs: Number of runs
            
        Returns:
            ExperimentResults
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*60}")
        
        results = ExperimentResults(name)
        
        for run_id in range(num_runs):
            logger.info(f"  Run {run_id + 1}/{num_runs}")
            
            # Set seed
            seed = config.get('training', {}).get('seed', 42) + run_id
            np.random.seed(seed)
            
            run_config = deepcopy(config)
            run_config['training']['seed'] = seed
            
            # Initialize LSED
            lsed = LSED(run_config, use_mock_llm=self.use_mock_llm)
            
            # Check if we should use LLM
            use_llm = run_config.get('ablation', {}).get('use_llm', True)
            
            n_clusters = len(np.unique(labels))
            
            # Summarize if enabled
            if use_llm:
                processed_texts = lsed.summarize_messages(texts, show_progress=False)
            else:
                processed_texts = texts
            
            # Vectorize
            vectors = lsed.vectorize_messages(processed_texts, timestamps)
            
            # Check if we should use hyperbolic encoding
            use_hyperbolic = run_config.get('ablation', {}).get('use_hyperbolic', True)
            
            if use_hyperbolic:
                # Split and train
                n = len(vectors)
                train_idx = int(n * 0.7)
                val_idx = int(n * 0.9)
                
                train_vectors = vectors[:train_idx]
                train_labels = labels[:train_idx]
                val_vectors = vectors[train_idx:val_idx]
                val_labels = labels[train_idx:val_idx]
                test_vectors = vectors[val_idx:]
                test_labels = labels[val_idx:]
                
                # Train
                lsed.train_model(
                    train_vectors, train_labels,
                    val_vectors, val_labels,
                    epochs=50,
                    verbose=False
                )
                
                # Encode and cluster
                embeddings = lsed.encode_hyperbolic(test_vectors, training=False)
                pred_labels = lsed.cluster_events(embeddings, n_clusters=n_clusters)
                
                metrics = ClusteringMetrics.compute_primary(test_labels, pred_labels)
            else:
                # Direct clustering without hyperbolic encoding
                n = len(vectors)
                test_idx = int(n * 0.8)
                test_vectors = vectors[test_idx:]
                test_labels = labels[test_idx:]
                
                pred_labels = lsed.clustering.fit_predict(test_vectors, n_clusters=n_clusters)
                metrics = ClusteringMetrics.compute_primary(test_labels, pred_labels)
            
            results.add_run(metrics)
        
        return results
    
    def ablation_vectorization(
        self,
        texts: List[str],
        timestamps: List,
        labels: np.ndarray
    ) -> Dict[str, ExperimentResults]:
        """
        Ablation study on vectorization methods.
        
        Compares:
        - SBERT vs Word2Vec
        - With vs without time vector
        
        Based on Table 3 in paper.
        """
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Vectorization Methods")
        logger.info("="*60)
        
        results = {}
        
        # Configurations to test
        configs = [
            ("LSED_W2V_wo_time", {"model": "word2vec", "include_time": False}),
            ("LSED_SBERT_wo_time", {"model": "sbert", "include_time": False}),
            ("LSED_W2V", {"model": "word2vec", "include_time": True}),
            ("LSED_SBERT", {"model": "sbert", "include_time": True}),
        ]
        
        for name, vec_config in configs:
            config = deepcopy(self.base_config)
            config['vectorization']['model'] = vec_config['model']
            config['vectorization']['include_time'] = vec_config['include_time']
            
            result = self.run_single_config(
                config, texts, timestamps, labels, name, num_runs=5
            )
            results[name] = result
            self.all_results[name] = result
        
        # Print comparison
        self._print_comparison(
            results,
            "Vectorization Ablation",
            ["LSED_W2V_wo_time", "LSED_SBERT_wo_time", "LSED_W2V", "LSED_SBERT"]
        )
        
        return results
    
    def ablation_hyperbolic(
        self,
        texts: List[str],
        timestamps: List,
        labels: np.ndarray
    ) -> Dict[str, ExperimentResults]:
        """
        Ablation study on hyperbolic embedding.
        
        Compares:
        - Without LLM & without Hyperbolic
        - Without LLM
        - Without Hyperbolic
        - Hyperboloid model
        - Poincaré model (full LSED)
        
        Based on Table 4 in paper.
        """
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Hyperbolic Embedding")
        logger.info("="*60)
        
        results = {}
        
        # Configurations to test
        configs = [
            ("LSED_wo_LLM_H", {"use_llm": False, "use_hyperbolic": False}),
            ("LSED_wo_LLM", {"use_llm": False, "use_hyperbolic": True}),
            ("LSED_wo_H", {"use_llm": True, "use_hyperbolic": False}),
            ("LSED_Hyperboloid", {"use_llm": True, "use_hyperbolic": True, "model": "hyperboloid"}),
            ("LSED_Poincare", {"use_llm": True, "use_hyperbolic": True, "model": "poincare"}),
        ]
        
        for name, ablation_config in configs:
            config = deepcopy(self.base_config)
            config['ablation'] = ablation_config
            
            if 'model' in ablation_config:
                config['hyperbolic']['model'] = ablation_config['model']
            
            result = self.run_single_config(
                config, texts, timestamps, labels, name, num_runs=5
            )
            results[name] = result
            self.all_results[name] = result
        
        # Print comparison
        self._print_comparison(
            results,
            "Hyperbolic Embedding Ablation",
            list(results.keys())
        )
        
        return results
    
    def ablation_prompt_type(
        self,
        texts: List[str],
        timestamps: List,
        labels: np.ndarray
    ) -> Dict[str, ExperimentResults]:
        """
        Ablation study on prompt types.
        
        Compares:
        - Summarize prompt
        - Paraphrase prompt
        
        Based on Table 8 in paper.
        """
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Prompt Types")
        logger.info("="*60)
        
        results = {}
        
        # Configurations to test
        prompt_types = ["summarize", "paraphrase"]
        
        for prompt_type in prompt_types:
            name = f"LSED_{prompt_type.capitalize()}"
            
            config = deepcopy(self.base_config)
            config['prompts']['prompt_type'] = prompt_type
            
            result = self.run_single_config(
                config, texts, timestamps, labels, name, num_runs=5
            )
            results[name] = result
            self.all_results[name] = result
        
        # Print comparison
        self._print_comparison(
            results,
            "Prompt Type Ablation",
            list(results.keys())
        )
        
        return results
    
    def ablation_hidden_layers(
        self,
        texts: List[str],
        timestamps: List,
        labels: np.ndarray
    ) -> Dict[str, ExperimentResults]:
        """
        Ablation study on number of hidden layers.
        
        Tests layers: 1, 2, 3, 4, 5
        
        Based on Figure 5(a) in paper.
        """
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Hidden Layers")
        logger.info("="*60)
        
        results = {}
        
        for num_layers in [1, 2, 3, 4, 5]:
            name = f"LSED_layers_{num_layers}"
            
            config = deepcopy(self.base_config)
            config['hyperbolic']['num_hidden_layers'] = num_layers
            
            result = self.run_single_config(
                config, texts, timestamps, labels, name, num_runs=5
            )
            results[name] = result
            self.all_results[name] = result
        
        # Print comparison
        self._print_comparison(
            results,
            "Hidden Layers Ablation",
            list(results.keys())
        )
        
        return results
    
    def ablation_hidden_dim(
        self,
        texts: List[str],
        timestamps: List,
        labels: np.ndarray
    ) -> Dict[str, ExperimentResults]:
        """
        Ablation study on hidden dimensions.
        
        Tests dimensions: 16, 32, 64, 128, 256, 512
        
        Based on Figure 5(b) in paper.
        """
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Hidden Dimensions")
        logger.info("="*60)
        
        results = {}
        
        for hidden_dim in [16, 32, 64, 128, 256, 512]:
            name = f"LSED_dim_{hidden_dim}"
            
            config = deepcopy(self.base_config)
            config['hyperbolic']['hidden_dim'] = hidden_dim
            
            result = self.run_single_config(
                config, texts, timestamps, labels, name, num_runs=5
            )
            results[name] = result
            self.all_results[name] = result
        
        # Print comparison
        self._print_comparison(
            results,
            "Hidden Dimension Ablation",
            list(results.keys())
        )
        
        return results
    
    def run_all(self) -> Dict[str, ExperimentResults]:
        """
        Run all ablation studies.
        
        Returns:
            Dictionary of all results
        """
        logger.info("\n" + "="*60)
        logger.info("RUNNING ALL ABLATION STUDIES")
        logger.info("="*60)
        
        # Load data
        texts, timestamps, labels = self.load_data()
        logger.info(f"Data loaded: {len(texts)} samples, {len(np.unique(labels))} events")
        
        # Run ablations
        self.ablation_vectorization(texts, timestamps, labels)
        self.ablation_hyperbolic(texts, timestamps, labels)
        self.ablation_prompt_type(texts, timestamps, labels)
        self.ablation_hidden_layers(texts, timestamps, labels)
        self.ablation_hidden_dim(texts, timestamps, labels)
        
        # Save all results
        self._save_results()
        
        return self.all_results
    
    def _print_comparison(
        self,
        results: Dict[str, ExperimentResults],
        title: str,
        order: List[str]
    ):
        """Print comparison table."""
        print("\n" + "="*60)
        print(title)
        print("="*60)
        
        print(f"\n{'Configuration':<25} {'NMI':>15} {'AMI':>15}")
        print("-"*55)
        
        for name in order:
            if name in results:
                result = results[name]
                nmi_mean, nmi_std = result.get_mean_std('nmi')
                ami_mean, ami_std = result.get_mean_std('ami')
                
                print(f"{name:<25} {nmi_mean:.2f}±{nmi_std:.2f}{' ':>3} {ami_mean:.2f}±{ami_std:.2f}")
        
        print("="*60)
    
    def _save_results(self):
        """Save all results to file."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }
        
        for name, result in self.all_results.items():
            nmi_mean, nmi_std = result.get_mean_std('nmi')
            ami_mean, ami_std = result.get_mean_std('ami')
            
            summary['results'][name] = {
                'nmi': {'mean': float(nmi_mean), 'std': float(nmi_std)},
                'ami': {'mean': float(ami_mean), 'std': float(ami_std)}
            }
        
        output_path = self.results_dir / 'ablation_results.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def run_ablation_study(
    config_path: str,
    data_path: str,
    use_mock_llm: bool = False
):
    """
    Convenience function to run ablation study.
    
    Args:
        config_path: Path to config file
        data_path: Path to data CSV
        use_mock_llm: Whether to use mock LLM
    """
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run ablation study
    study = AblationStudy(
        config=config,
        data_path=data_path,
        use_mock_llm=use_mock_llm
    )
    
    results = study.run_all()
    
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
    parser = argparse.ArgumentParser(description="Run LSED Ablation Study")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default="data/sample_data.csv")
    parser.add_argument("--mock-llm", action="store_true")
    args = parser.parse_args()
    
    # Create sample data if needed
    from data.data_loader import create_sample_data
    if not Path(args.data).exists():
        create_sample_data(args.data, n_messages=500, n_events=20)
    
    # Run ablation study
    results = run_ablation_study(
        config_path=args.config,
        data_path=args.data,
        use_mock_llm=args.mock_llm
    )
