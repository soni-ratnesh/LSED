"""
LSED Model - Main Framework
LLM-enhanced Social Event Detection

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

This module implements the complete LSED pipeline:
1. LLM Summarization - Expand abbreviations and standardize expressions
2. Vectorization - SBERT + Time encoding
3. Hyperbolic Encoding - Project to hyperbolic space
4. Clustering - K-Means for event detection
5. Incremental Learning - Window-based updates
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Import LSED modules
from models.llm_summarizer import LLMSummarizer
from models.vectorizer import Vectorizer
from models.hyperbolic_encoder import HyperbolicEncoder
from models.clustering import EventClustering
from utils.metrics import ClusteringMetrics, ExperimentResults

logger = logging.getLogger(__name__)


class LSED(nn.Module):
    """
    LLM-enhanced Social Event Detection Framework.
    
    Complete pipeline for detecting social events from message streams:
    
    Step 1: Prompt LLM to summarize social messages
    Step 2: Vectorize the summarized messages and timestamps
    Step 3: Project vectors into hyperbolic space
    Step 4: Maintenance and event detection (clustering)
    
    The framework supports both offline and online (incremental) scenarios.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        use_mock_llm: bool = False
    ):
        """
        Initialize LSED framework.
        
        Args:
            config: Configuration dictionary
            use_mock_llm: Whether to use mock LLM for testing
        """
        super().__init__()
        
        self.config = config
        self.use_mock_llm = use_mock_llm
        
        # Training settings
        self.training_config = config.get('training', {})
        self.lr = self.training_config.get('learning_rate', 0.01)
        self.epochs = self.training_config.get('epochs', 100)
        self.batch_size = self.training_config.get('batch_size', 256)
        self.patience = self.training_config.get('patience', 10)
        self.seed = self.training_config.get('seed', 42)
        
        # Incremental settings
        self.incremental_config = config.get('incremental', {})
        self.window_size = self.incremental_config.get('window_size', 1)
        
        # Set random seed
        self._set_seed(self.seed)
        
        # Initialize components
        self._init_components()
        
        # Device setup
        self.device = self._get_device()
        
        # Move encoder to device
        if hasattr(self, 'hyperbolic_encoder'):
            self.hyperbolic_encoder = self.hyperbolic_encoder.to(self.device)
        
        logger.info(f"LSED initialized on device: {self.device}")
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _get_device(self) -> torch.device:
        """Get compute device."""
        device_config = self.config.get('device', {})
        use_gpu = device_config.get('use_gpu', True)
        gpu_id = device_config.get('gpu_id', 0)
        
        if use_gpu and torch.cuda.is_available():
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cpu')
    
    def _init_components(self):
        """Initialize all LSED components."""
        # Step 1: LLM Summarizer
        logger.info("Initializing LLM Summarizer...")
        self.summarizer = LLMSummarizer(self.config, use_mock=self.use_mock_llm)
        
        # Step 2: Vectorizer
        logger.info("Initializing Vectorizer...")
        self.vectorizer = Vectorizer(self.config)
        
        # Step 3: Hyperbolic Encoder
        logger.info("Initializing Hyperbolic Encoder...")
        input_dim = self.vectorizer.get_embedding_dim()
        self.hyperbolic_encoder = HyperbolicEncoder(self.config, input_dim=input_dim)
        
        # Step 4: Clustering
        logger.info("Initializing Clustering...")
        self.clustering = EventClustering(self.config)
        
        # Loss function for supervised training
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.hyperbolic_encoder.parameters(),
            lr=self.lr,
            weight_decay=self.training_config.get('weight_decay', 0.0001)
        )
    
    def summarize_messages(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """
        Step 1: Summarize social messages using LLM.
        
        Args:
            texts: List of original social messages
            show_progress: Whether to show progress bar
            
        Returns:
            List of summarized messages
        """
        logger.info(f"Summarizing {len(texts)} messages...")
        start_time = time.time()
        
        summarized = self.summarizer.get_summarized_texts(
            texts,
            show_progress=show_progress
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Summarization completed in {elapsed:.2f}s")
        
        return summarized
    
    def vectorize_messages(
        self,
        texts: List[str],
        timestamps: Optional[List] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Step 2: Vectorize messages using SBERT + time encoding.
        
        Args:
            texts: List of (summarized) messages
            timestamps: Optional list of timestamps
            show_progress: Whether to show progress bar
            
        Returns:
            Message vectors (n_samples, embedding_dim)
        """
        logger.info(f"Vectorizing {len(texts)} messages...")
        
        vectors = self.vectorizer.vectorize_batch(
            texts,
            timestamps=timestamps,
            show_progress=show_progress
        )
        
        logger.info(f"Vectorization complete: shape={vectors.shape}")
        return vectors
    
    def encode_hyperbolic(
        self,
        vectors: np.ndarray,
        training: bool = False
    ) -> np.ndarray:
        """
        Step 3: Encode vectors into hyperbolic space.
        
        Args:
            vectors: Message vectors (Euclidean)
            training: Whether in training mode
            
        Returns:
            Hyperbolic embeddings
        """
        # Convert to tensor
        x = torch.tensor(vectors, dtype=torch.float32).to(self.device)
        
        # Encode
        self.hyperbolic_encoder.train(training)
        with torch.set_grad_enabled(training):
            embeddings = self.hyperbolic_encoder(x)
        
        return embeddings.detach().cpu().numpy()
    
    def cluster_events(
        self,
        embeddings: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """
        Step 4: Cluster embeddings into events.
        
        Args:
            embeddings: Hyperbolic embeddings
            n_clusters: Number of event clusters
            
        Returns:
            Cluster labels
        """
        logger.info(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters...")
        
        labels = self.clustering.fit_predict(embeddings, n_clusters=n_clusters)
        
        return labels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hyperbolic encoder.
        
        Args:
            x: Input vectors
            
        Returns:
            Hyperbolic embeddings
        """
        return self.hyperbolic_encoder(x)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Training metrics
        """
        self.hyperbolic_encoder.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            embeddings = self.hyperbolic_encoder(batch_x)
            
            # Get Euclidean embeddings for classification
            euclidean = self.hyperbolic_encoder.encoder.get_euclidean_embeddings(embeddings)
            
            # Classifier should already be initialized in train_model()
            if not hasattr(self, 'classifier') or self.classifier is None:
                raise RuntimeError("Classifier not initialized. Call train_model() instead of train_epoch() directly.")
            
            logits = self.classifier(euclidean)
            loss = self.criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return {'loss': total_loss / n_batches}
    
    def train_model(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        val_vectors: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the LSED model.
        
        Args:
            vectors: Training vectors
            labels: Training labels
            val_vectors: Validation vectors
            val_labels: Validation labels
            epochs: Number of epochs
            verbose: Whether to show progress
            
        Returns:
            Training history
        """
        epochs = epochs or self.epochs
        
        # Encode labels to contiguous values [0, n_classes-1]
        # This is necessary because CrossEntropyLoss expects labels in this range
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        n_classes = len(self.label_encoder.classes_)
        
        # Also encode validation labels if provided
        encoded_val_labels = None
        if val_labels is not None:
            # Handle potential unseen labels in validation set
            # Fit a combined encoder or just use transform with error handling
            all_unique_labels = np.unique(np.concatenate([labels, val_labels]))
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_unique_labels)
            encoded_labels = self.label_encoder.transform(labels)
            encoded_val_labels = self.label_encoder.transform(val_labels)
            n_classes = len(self.label_encoder.classes_)
        
        # Prepare data
        train_dataset = TensorDataset(
            torch.tensor(vectors, dtype=torch.float32),
            torch.tensor(encoded_labels, dtype=torch.long)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Initialize classifier with correct number of classes
        self.classifier = nn.Linear(
            self.config.get('hyperbolic', {}).get('hidden_dim', 64),
            n_classes
        ).to(self.device)
        
        # Optimizer for classifier too
        optimizer = optim.Adam(
            list(self.hyperbolic_encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.lr,
            weight_decay=self.training_config.get('weight_decay', 0.0001)
        )
        
        history = {'train_loss': [], 'val_nmi': [], 'val_ami': []}
        best_nmi = 0.0
        patience_counter = 0
        
        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training")
        
        for epoch in iterator:
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            if val_vectors is not None and encoded_val_labels is not None:
                embeddings = self.encode_hyperbolic(val_vectors, training=False)
                pred_labels = self.cluster_events(embeddings, n_clusters=len(np.unique(encoded_val_labels)))
                
                val_metrics = ClusteringMetrics.compute_primary(encoded_val_labels, pred_labels)
                history['val_nmi'].append(val_metrics['nmi'])
                history['val_ami'].append(val_metrics['ami'])
                
                # Early stopping
                if val_metrics['nmi'] > best_nmi:
                    best_nmi = val_metrics['nmi']
                    patience_counter = 0
                    # Save best model
                    self.best_state = {
                        'encoder': self.hyperbolic_encoder.state_dict(),
                        'classifier': self.classifier.state_dict()
                    }
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if verbose:
                    iterator.set_postfix({
                        'loss': train_metrics['loss'],
                        'nmi': val_metrics['nmi']
                    })
        
        return history
    
    def predict(
        self,
        texts: List[str],
        timestamps: Optional[List] = None,
        n_clusters: int = None,
        summarize: bool = True
    ) -> np.ndarray:
        """
        Full prediction pipeline.
        
        Args:
            texts: Input texts
            timestamps: Optional timestamps
            n_clusters: Number of clusters
            summarize: Whether to summarize texts first
            
        Returns:
            Predicted event labels
        """
        # Step 1: Summarize (optional)
        if summarize:
            texts = self.summarize_messages(texts, show_progress=False)
        
        # Step 2: Vectorize
        vectors = self.vectorize_messages(texts, timestamps)
        
        # Step 3: Hyperbolic encoding
        embeddings = self.encode_hyperbolic(vectors, training=False)
        
        # Step 4: Cluster
        if n_clusters is None:
            raise ValueError("n_clusters must be specified")
        
        labels = self.cluster_events(embeddings, n_clusters)
        
        return labels
    
    def evaluate(
        self,
        texts: List[str],
        timestamps: Optional[List],
        true_labels: np.ndarray,
        summarize: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            texts: Input texts
            timestamps: Timestamps
            true_labels: Ground truth labels
            summarize: Whether to summarize texts
            
        Returns:
            Evaluation metrics
        """
        n_clusters = len(np.unique(true_labels))
        
        pred_labels = self.predict(
            texts,
            timestamps=timestamps,
            n_clusters=n_clusters,
            summarize=summarize
        )
        
        metrics = ClusteringMetrics.compute_primary(true_labels, pred_labels)
        
        return metrics
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config,
            'encoder_state': self.hyperbolic_encoder.state_dict(),
            'classifier_state': self.classifier.state_dict() if hasattr(self, 'classifier') else None
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.hyperbolic_encoder.load_state_dict(checkpoint['encoder_state'])
        
        if checkpoint['classifier_state'] is not None and hasattr(self, 'classifier'):
            self.classifier.load_state_dict(checkpoint['classifier_state'])
        
        logger.info(f"Model loaded from {path}")


class IncrementalLSED(LSED):
    """
    Incremental LSED for online social event detection.
    
    Implements window-based model updates as described in the paper.
    
    Algorithm:
    - If i % w == 0: Update model parameters (train)
    - Else: Just run inference
    
    Where:
    - i is the message block index
    - w is the window size
    """
    
    def __init__(self, config: Dict[str, Any], use_mock_llm: bool = False):
        """Initialize Incremental LSED."""
        super().__init__(config, use_mock_llm)
        
        self.block_counter = 0
        self.history: Dict[str, List] = {'nmi': [], 'ami': [], 'blocks': []}
    
    def should_update(self) -> bool:
        """Check if model should be updated based on window size."""
        return self.block_counter % self.window_size == 0
    
    def process_block(
        self,
        block_name: str,
        texts: List[str],
        timestamps: List,
        true_labels: np.ndarray,
        summarize: bool = True,
        train: bool = True
    ) -> Dict[str, float]:
        """
        Process a single message block.
        
        Args:
            block_name: Block identifier (e.g., "M1")
            texts: Message texts
            timestamps: Timestamps
            true_labels: Ground truth labels
            summarize: Whether to summarize
            train: Whether to train (if window allows)
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Processing block {block_name} with {len(texts)} messages")
        
        n_clusters = len(np.unique(true_labels))
        
        # Summarize
        if summarize:
            texts = self.summarize_messages(texts, show_progress=True)
        
        # Vectorize
        vectors = self.vectorize_messages(texts, timestamps)
        
        # Check if we should update
        if train and self.should_update():
            logger.info(f"Updating model at block {block_name} (counter={self.block_counter})")
            
            # Split for training
            n = len(vectors)
            train_idx = int(n * 0.7)
            val_idx = int(n * 0.9)
            
            train_vectors = vectors[:train_idx]
            train_labels = true_labels[:train_idx]
            val_vectors = vectors[train_idx:val_idx]
            val_labels = true_labels[train_idx:val_idx]
            
            # Train
            self.train_model(
                train_vectors, train_labels,
                val_vectors, val_labels,
                epochs=self.epochs,
                verbose=True
            )
        
        # Evaluate
        embeddings = self.encode_hyperbolic(vectors, training=False)
        pred_labels = self.cluster_events(embeddings, n_clusters)
        
        metrics = ClusteringMetrics.compute_primary(true_labels, pred_labels)
        
        # Store history
        self.history['nmi'].append(metrics['nmi'])
        self.history['ami'].append(metrics['ami'])
        self.history['blocks'].append(block_name)
        
        self.block_counter += 1
        
        logger.info(f"Block {block_name}: NMI={metrics['nmi']:.4f}, AMI={metrics['ami']:.4f}")
        
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all processed blocks."""
        return {
            'avg_nmi': np.mean(self.history['nmi']),
            'avg_ami': np.mean(self.history['ami']),
            'std_nmi': np.std(self.history['nmi']),
            'std_ami': np.std(self.history['ami'])
        }


if __name__ == "__main__":
    # Test LSED model
    import yaml
    
    print("Testing LSED Model")
    print("="*60)
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize with mock LLM
    print("\n1. Initializing LSED...")
    lsed = LSED(config, use_mock_llm=True)
    
    # Test with synthetic data
    print("\n2. Testing with synthetic data...")
    
    np.random.seed(42)
    n_samples = 100
    n_events = 5
    
    texts = [f"Sample message {i} about event {i % n_events}" for i in range(n_samples)]
    timestamps = [f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00" for i in range(n_samples)]
    true_labels = np.array([i % n_events for i in range(n_samples)])
    
    print(f"   Samples: {n_samples}, Events: {n_events}")
    
    # Test summarization
    print("\n3. Testing summarization...")
    summarized = lsed.summarize_messages(texts[:5], show_progress=False)
    for orig, summ in zip(texts[:3], summarized[:3]):
        print(f"   {orig[:30]}... -> {summ[:30]}...")
    
    # Test vectorization
    print("\n4. Testing vectorization...")
    vectors = lsed.vectorize_messages(summarized, timestamps[:5])
    print(f"   Vector shape: {vectors.shape}")
    
    # Test hyperbolic encoding
    print("\n5. Testing hyperbolic encoding...")
    embeddings = lsed.encode_hyperbolic(vectors)
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Embedding norms: {np.linalg.norm(embeddings, axis=1)}")
    
    # Test clustering
    print("\n6. Testing clustering...")
    pred_labels = lsed.cluster_events(embeddings, n_clusters=n_events)
    print(f"   Predicted labels: {pred_labels}")
    
    # Test full pipeline
    print("\n7. Testing full prediction pipeline...")
    pred_labels = lsed.predict(texts, timestamps, n_clusters=n_events, summarize=True)
    print(f"   Predictions shape: {pred_labels.shape}")
    
    # Evaluate
    print("\n8. Evaluating...")
    metrics = ClusteringMetrics.compute_primary(true_labels, pred_labels)
    print(f"   NMI: {metrics['nmi']:.4f}")
    print(f"   AMI: {metrics['ami']:.4f}")
    
    # Test incremental
    print("\n9. Testing Incremental LSED...")
    inc_lsed = IncrementalLSED(config, use_mock_llm=True)
    
    # Process multiple blocks
    for i in range(3):
        block_texts = texts[i*30:(i+1)*30]
        block_times = timestamps[i*30:(i+1)*30]
        block_labels = true_labels[i*30:(i+1)*30]
        
        metrics = inc_lsed.process_block(
            f"M{i}",
            block_texts,
            block_times,
            block_labels,
            train=(i == 0)
        )
    
    avg_metrics = inc_lsed.get_average_metrics()
    print(f"\n   Average NMI: {avg_metrics['avg_nmi']:.4f} ± {avg_metrics['std_nmi']:.4f}")
    print(f"   Average AMI: {avg_metrics['avg_ami']:.4f} ± {avg_metrics['std_ami']:.4f}")
    
    print("\n" + "="*60)
    print("All tests passed!")