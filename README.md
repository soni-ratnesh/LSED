# LSED: LLM-enhanced Social Event Detection

Implementation of the paper: **"Text is All You Need: LLM-enhanced Incremental Social Event Detection"** (ACL 2025)

## Overview

Social Event Detection (SED) is the task of identifying, categorizing, and tracking events from social data sources such as social media posts. This implementation provides a novel framework that leverages Large Language Models (LLMs) to address the key challenges in SED:

- **Informal expressions and abbreviations** in short social media texts
- **Hierarchical structure** of natural language sentences
- **Dynamic nature** of social message streams

## Key Features

- **LLM Summarization**: Uses LLMs (Llama3.1, Qwen2.5, Gemma2) to expand abbreviations and standardize textual expressions
- **Hyperbolic Embeddings**: Projects text representations into hyperbolic space to better capture hierarchical structures
- **Incremental Learning**: Supports online/streaming scenarios with window-based model updates
- **Modular Design**: Each component can be used independently or replaced

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LSED Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  Step 1: LLM Summarization                                  │
│  ┌─────────┐    ┌─────────────────────────────────────┐    │
│  │ Message │───▶│ LLM (Llama3.1/Qwen2.5/Gemma2)       │    │
│  │ Block   │    │ - Expand abbreviations              │    │
│  └─────────┘    │ - Standardize expressions           │    │
│                 └─────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Vectorization                                      │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Summarized │  │ SBERT/W2V    │  │ + Time Encoding    │  │
│  │ Text       │─▶│ Embeddings   │─▶│ (OLE Date Format)  │  │
│  └────────────┘  └──────────────┘  └────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Step 3: Hyperbolic Encoding                                │
│  ┌───────────┐   ┌─────────────────────────────────────┐   │
│  │ Euclidean │──▶│ Poincaré Ball / Hyperboloid Model   │   │
│  │ Vectors   │   │ (3 hidden layers, dim=64)           │   │
│  └───────────┘   └─────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Clustering                                         │
│  ┌─────────────┐   ┌─────────────────────────────────┐     │
│  │ Hyperbolic  │──▶│ K-Means Clustering              │     │
│  │ Embeddings  │   │ ──▶ Event Labels                │     │
│  └─────────────┘   └─────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/soni-ratnesh/LSED
cd LSED

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install spaCy model for Word2Vec
python -m spacy download en_core_web_sm
```

### LLM Setup (Ollama)

LSED uses Ollama for running local LLMs. Install Ollama and pull the required models:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull gemma2:9b
```

## Data Format

Input CSV should have the following columns:
- `text`: The social message content
- `created_at`: Timestamp (ISO format)
- `event`: Ground truth event label

Example:
```csv
text,created_at,event
"Just saw the news about the earthquake!",2024-01-15T10:30:00,earthquake_2024
"OMG the NBA finals were amazing last night",2024-01-15T11:00:00,nba_finals
```

## Datasets

### Synthetic Data (Included)

For testing purposes, synthetic data is provided in `data/synthetic_data.csv` (5,000 messages, 50 events).

```bash
python main.py --mode offline --data data/synthetic_data.csv --mock-llm
```

### Twitter Event Datasets (Event2012, Event2018)

1. Download tweet IDs from [Twitter event datasets (2012-2016) - Figshare](https://figshare.com/articles/dataset/Twitter_event_datasets_2012-2016_/5100460)

2. Rehydrate tweets using [twarc2](https://twarc-project.readthedocs.io/):
   ```bash
   pip install twarc
   twarc2 configure  # Setup Twitter API credentials
   twarc2 hydrate tweet_ids.txt tweets.jsonl
   ```

### Twitter event 2012-16 Corpora (English/French)

For downloading event detection datasets, use: [Twitter event datasets (2012-2016)](https://figshare.com/articles/dataset/Twitter_event_datasets_2012-2016_/5100460))

## Usage

### Command Line

```bash
# Run offline experiment
python main.py --mode offline --data data/events.csv

# Run online (incremental) experiment
python main.py --mode online --data data/events.csv --window-size 1

# Run ablation study
python main.py --mode ablation --data data/events.csv

# Run all experiments
python main.py --mode all --data data/events.csv

# Test without real LLM (mock mode)
python main.py --mode offline --data data/sample.csv --mock-llm --create-sample
```

### Python API

```python
from lsed import LSED
from config import load_config

# Load configuration
config = load_config("config/config.yaml")

# Initialize LSED
model = LSED(config, use_mock_llm=False)

# Predict events
texts = ["Message 1", "Message 2", "Message 3"]
timestamps = ["2024-01-01T10:00:00", "2024-01-01T11:00:00", "2024-01-01T12:00:00"]

predictions = model.predict(texts, timestamps, n_clusters=5)
```

### Incremental Learning

```python
from lsed import IncrementalLSED
from data import DataLoader

# Load data
loader = DataLoader(config)
loader.load_data("data/events.csv")
offline_block, online_blocks = loader.split_into_blocks()

# Initialize incremental model
model = IncrementalLSED(config)

# Process offline block (M0)
model.process_block("M0", texts_0, timestamps_0, labels_0, train=True)

# Process online blocks (M1, M2, ...)
for block in online_blocks:
    metrics = model.process_block(
        block.block_name,
        block_texts,
        block_timestamps,
        block_labels,
        train=model.should_update()
    )
```

## Configuration

All hyperparameters are configured in `config/config.yaml`:

```yaml
# LLM Configuration
llm:
  model: "llama3.1"  # Options: llama3.1, qwen2.5, gemma2
  temperature: 0.1
  max_tokens: 256

# Prompt Configuration
prompts:
  prompt_type: "summarize"  # Options: summarize, paraphrase

# Vectorization
vectorization:
  model: "sbert"  # Options: sbert, word2vec
  sbert_model: "all-MiniLM-L6-v2"
  include_time: true

# Hyperbolic Encoder
hyperbolic:
  model: "poincare"  # Options: poincare, hyperboloid
  curvature: 1.0
  num_hidden_layers: 3
  hidden_dim: 64

# Training
training:
  learning_rate: 0.01
  epochs: 100
  batch_size: 256

# Incremental
incremental:
  window_size: 1
```

## Project Structure

```
lsed/
├── config/
│   ├── __init__.py
│   └── config.yaml           # Main configuration file
├── data/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── generate_synthetic.py # Synthetic data generator
│   ├── synthetic_data.csv    # Pre-generated synthetic dataset (5K messages, 50 events)
│   └── sample_data.csv       # Small sample for quick testing
├── models/
│   ├── __init__.py
│   ├── lsed.py               # Main LSED model
│   ├── llm_summarizer.py     # LLM summarization
│   ├── vectorizer.py         # Text vectorization
│   ├── hyperbolic_encoder.py # Hyperbolic embeddings
│   └── clustering.py         # K-Means clustering
├── utils/
│   ├── __init__.py
│   └── metrics.py            # Evaluation metrics (NMI, AMI)
├── experiments/
│   ├── __init__.py
│   ├── offline_experiment.py # Offline scenario
│   ├── online_experiment.py  # Online/incremental scenario
│   └── ablation_study.py     # Ablation studies
├── results/                  # Experiment results (auto-generated)
│   ├── offline/
│   └── online/
├── main.py                   # Main runner script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Experiments

### Offline Scenario

Train and evaluate on the first 7 days of data (M0):

```bash
python main.py --mode offline --data data/events.csv --num-runs 5
```

### Online Scenario

Process message blocks incrementally with window-based updates:

```bash
python main.py --mode online --data data/events.csv --window-size 1
```

### Ablation Studies

Evaluate impact of different components:

1. **Vectorization**: SBERT vs Word2Vec, with/without time
2. **Hyperbolic Embedding**: Poincaré vs Hyperboloid vs Euclidean
3. **LLM Summarization**: With vs without LLM
4. **Prompt Types**: Summarize vs Paraphrase
5. **Hidden Layers**: 1-5 layers
6. **Hidden Dimensions**: 16-512

```bash
python main.py --mode ablation --data data/events.csv
```

## Evaluation Metrics

- **NMI** (Normalized Mutual Information): Measures mutual dependence between true and predicted labels
- **AMI** (Adjusted Mutual Information): NMI adjusted for chance

Results are reported as mean ± std over 5 runs.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{qiu2025lsed,
  title={Text is All You Need: LLM-enhanced Incremental Social Event Detection},
  author={Qiu, Zitai and Ma, Congbo and Wu, Jia and Yang, Jian},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  pages={4666--4680},
  year={2025}
}
```

## License

This project is for research purposes. Please refer to the original paper for more details.

## Acknowledgments

- Based on the ACL 2025 paper by Qiu et al.
- Uses sentence-transformers for SBERT embeddings
- Uses Ollama for local LLM inference
