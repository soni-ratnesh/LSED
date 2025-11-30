"""
LSED - LLM-enhanced Social Event Detection

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
Qiu et al., ACL 2025

This framework leverages Large Language Models and hyperbolic embeddings
for effective, efficient, and stable social event detection.

Main Components:
- LLM Summarizer: Expands abbreviations and standardizes expressions
- Vectorizer: SBERT/Word2Vec + time encoding
- Hyperbolic Encoder: Projects embeddings to hyperbolic space
- Clustering: K-Means for event detection

Usage:
    from lsed import LSED
    
    model = LSED(config)
    predictions = model.predict(texts, timestamps, n_clusters=10)
"""

__version__ = "1.0.0"
__author__ = "Based on Qiu et al., ACL 2025"

from models.lsed import LSED, IncrementalLSED
from data.data_loader import DataLoader, MessageBlock
from utils.metrics import ClusteringMetrics, ExperimentResults

__all__ = [
    "LSED",
    "IncrementalLSED",
    "DataLoader",
    "MessageBlock",
    "ClusteringMetrics",
    "ExperimentResults",
]
