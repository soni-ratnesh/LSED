#!/usr/bin/env python
"""
LSED - LLM-enhanced Social Event Detection
Main Runner Script

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

Usage:
    python main.py --mode offline --data data/events.csv
    python main.py --mode online --data data/events.csv
    python main.py --mode ablation --data data/events.csv
    python main.py --mode all --data data/events.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader, create_sample_data
from experiments.offline_experiment import OfflineExperiment
from experiments.online_experiment import OnlineExperiment, WindowSizeAnalysis
from experiments.ablation_study import AblationStudy


def setup_logging(log_file: str = None, level: str = "INFO"):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_offline(config: dict, data_path: str, use_mock_llm: bool = False):
    """Run offline experiment."""
    print("\n" + "="*60)
    print("LSED - OFFLINE EXPERIMENT")
    print("="*60)
    
    experiment = OfflineExperiment(
        config=config,
        data_path=data_path,
        results_dir="results/offline",
        use_mock_llm=use_mock_llm
    )
    
    results = experiment.run()
    return results


def run_online(config: dict, data_path: str, use_mock_llm: bool = False):
    """Run online experiment."""
    print("\n" + "="*60)
    print("LSED - ONLINE EXPERIMENT")
    print("="*60)
    
    experiment = OnlineExperiment(
        config=config,
        data_path=data_path,
        results_dir="results/online",
        use_mock_llm=use_mock_llm
    )
    
    results, block_results = experiment.run()
    return results, block_results


def run_ablation(config: dict, data_path: str, use_mock_llm: bool = False):
    """Run ablation study."""
    print("\n" + "="*60)
    print("LSED - ABLATION STUDY")
    print("="*60)
    
    study = AblationStudy(
        config=config,
        data_path=data_path,
        results_dir="results/ablation",
        use_mock_llm=use_mock_llm
    )
    
    results = study.run_all()
    return results


def run_window_analysis(config: dict, data_path: str, use_mock_llm: bool = False):
    """Run window size analysis."""
    print("\n" + "="*60)
    print("LSED - WINDOW SIZE ANALYSIS")
    print("="*60)
    
    analysis = WindowSizeAnalysis(
        config=config,
        data_path=data_path,
        results_dir="results/window_analysis",
        use_mock_llm=use_mock_llm
    )
    
    results = analysis.run()
    return results


def run_all(config: dict, data_path: str, use_mock_llm: bool = False):
    """Run all experiments."""
    print("\n" + "="*60)
    print("LSED - RUNNING ALL EXPERIMENTS")
    print("="*60)
    
    all_results = {}
    
    # Offline
    all_results['offline'] = run_offline(config, data_path, use_mock_llm)
    
    # Online
    online_results, block_results = run_online(config, data_path, use_mock_llm)
    all_results['online'] = online_results
    all_results['online_blocks'] = block_results
    
    # Ablation
    all_results['ablation'] = run_ablation(config, data_path, use_mock_llm)
    
    # Window analysis
    all_results['window'] = run_window_analysis(config, data_path, use_mock_llm)
    
    return all_results


def print_final_summary(mode: str):
    """Print final summary of experiments."""
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nMode: {mode}")
    print(f"Results saved to: results/{mode}/")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LSED - LLM-enhanced Social Event Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode offline --data data/events.csv
  python main.py --mode online --data data/events.csv --window-size 1
  python main.py --mode ablation --data data/events.csv
  python main.py --mode all --data data/events.csv

For testing without real LLM:
  python main.py --mode offline --data data/sample.csv --mock-llm
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "online", "ablation", "window", "all"],
        default="offline",
        help="Experiment mode to run"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/input.csv",
        help="Path to input CSV file (text, created_at, event columns)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM for testing (no actual API calls)"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample data for testing"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Override number of experiment runs"
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override window size for online mode"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        choices=["llama3.1", "qwen2.5", "gemma2"],
        default=None,
        help="Override LLM model"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file, args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print header
    print("\n" + "="*60)
    print("LSED - LLM-enhanced Social Event Detection")
    print("Text is All You Need (ACL 2025)")
    print("="*60)
    
    # Create sample data if requested
    if args.create_sample:
        sample_path = "data/sample_data.csv"
        create_sample_data(sample_path, n_messages=1000, n_events=30)
        args.data = sample_path
        logger.info(f"Sample data created at {sample_path}")
    
    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Use --create-sample to create sample data for testing")
        sys.exit(1)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Override config with command line arguments
    if args.num_runs is not None:
        config['evaluation']['num_runs'] = args.num_runs
    
    if args.window_size is not None:
        config['incremental']['window_size'] = args.window_size
    
    if args.llm_model is not None:
        config['llm']['model'] = args.llm_model
    
    # Log configuration
    logger.info(f"Data: {args.data}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Mock LLM: {args.mock_llm}")
    logger.info(f"LLM Model: {config.get('llm', {}).get('model', 'llama3.1')}")
    
    # Run experiment
    try:
        if args.mode == "offline":
            run_offline(config, str(data_path), args.mock_llm)
        
        elif args.mode == "online":
            run_online(config, str(data_path), args.mock_llm)
        
        elif args.mode == "ablation":
            run_ablation(config, str(data_path), args.mock_llm)
        
        elif args.mode == "window":
            run_window_analysis(config, str(data_path), args.mock_llm)
        
        elif args.mode == "all":
            run_all(config, str(data_path), args.mock_llm)
        
        print_final_summary(args.mode)
        
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
