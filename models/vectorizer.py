"""
Vectorizer Module for LSED
Handles text vectorization using SBERT or Word2Vec, plus time encoding.

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

Key components:
1. SBERT (sentence-transformers) - preferred, contains more semantic info
2. Word2Vec (spaCy) - alternative, simpler approach
3. Time encoding using OLE date format

Paper findings:
- SBERT performs ~8% better than Word2Vec
- Adding time vector improves performance by ~2%
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TimeEncoder:
    """
    Time encoder using OLE (Object Linking and Embedding) date format.
    
    Based on KPGNN (Cao et al., 2021) time vectorization approach.
    
    Formula:
        TIME(t) = (t_days / D_max, t_seconds / S_max)
    
    Where:
        - t_days: days since epoch
        - t_seconds: seconds within the day
        - D_max: normalization factor for days (100000)
        - S_max: seconds in a day (86400)
    """
    
    def __init__(
        self, 
        d_max: int = 100000,
        s_max: int = 86400,
        reference_date: Optional[datetime] = None
    ):
        """
        Initialize time encoder.
        
        Args:
            d_max: Normalization factor for days (paper: 100000)
            s_max: Seconds in a day (86400)
            reference_date: Reference date for computing days (default: Unix epoch)
        """
        self.d_max = d_max
        self.s_max = s_max
        self.reference_date = reference_date or datetime(1970, 1, 1)
        
    def encode(self, timestamp: Union[datetime, str]) -> np.ndarray:
        """
        Encode a single timestamp.
        
        Args:
            timestamp: Datetime object or string
            
        Returns:
            2D vector [days_component, seconds_component]
        """
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Calculate days since reference
        delta = timestamp - self.reference_date
        t_days = delta.days
        
        # Calculate seconds within the day
        t_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        
        # Normalize to [0, 1]
        days_component = t_days / self.d_max
        seconds_component = t_seconds / self.s_max
        
        return np.array([days_component, seconds_component], dtype=np.float32)
    
    def encode_batch(self, timestamps: List[Union[datetime, str]]) -> np.ndarray:
        """
        Encode a batch of timestamps.
        
        Args:
            timestamps: List of datetime objects or strings
            
        Returns:
            Array of shape (n, 2)
        """
        return np.array([self.encode(t) for t in timestamps], dtype=np.float32)


class SBERTVectorizer:
    """
    SBERT-based text vectorizer using sentence-transformers.
    
    Uses the "all-MiniLM-L6-v2" model as specified in the paper.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        """
        Initialize SBERT vectorizer.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda' or 'cpu')
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            
            # Move to specified device if available
            if device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.model = self.model.to(device)
                        logger.info(f"SBERT model loaded on GPU")
                    else:
                        logger.info(f"CUDA not available, using CPU")
                except:
                    pass
            
            logger.info(f"SBERT model loaded: {model_name}")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Array of shape (n, embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim


class Word2VecVectorizer:
    """
    Word2Vec-based text vectorizer using spaCy.
    
    Uses the "en_core_web_sm" model as specified in the paper.
    Averages word vectors to get sentence vectors.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize Word2Vec vectorizer.
        
        Args:
            model_name: Name of the spaCy model
        """
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            logger.info(f"spaCy model loaded: {model_name}")
            
            # Get embedding dimension from first word
            self.embedding_dim = self.nlp.vocab.vectors_length
            if self.embedding_dim == 0:
                # Fallback for models without word vectors
                self.embedding_dim = 96  # en_core_web_sm uses 96
                logger.warning(f"Model has no vectors, using dim={self.embedding_dim}")
            
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except OSError:
            raise ImportError(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            )
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (average of word vectors)
        """
        doc = self.nlp(text)
        
        # Get word vectors and average them
        vectors = [token.vector for token in doc if token.has_vector]
        
        if vectors:
            return np.mean(vectors, axis=0).astype(np.float32)
        else:
            # Return zero vector if no words have vectors
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def encode_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of shape (n, embedding_dim)
        """
        return np.array([self.encode(text) for text in texts], dtype=np.float32)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim


class Vectorizer:
    """
    Main vectorizer class that combines text embeddings and time encoding.
    
    Supports both SBERT and Word2Vec backends.
    
    Final vector: v_j = ADD(v_ms_j, v_t_j)
    Where ADD concatenates the message vector and time vector.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vectorizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vec_config = config.get('vectorization', {})
        
        # Get settings
        self.model_type = self.vec_config.get('model', 'sbert')
        self.include_time = self.vec_config.get('include_time', True)
        
        # Initialize text vectorizer
        if self.model_type.lower() == 'sbert':
            sbert_model = self.vec_config.get('sbert_model', 'all-MiniLM-L6-v2')
            device = 'cuda' if config.get('device', {}).get('use_gpu', True) else 'cpu'
            self.text_vectorizer = SBERTVectorizer(model_name=sbert_model, device=device)
        elif self.model_type.lower() == 'word2vec':
            w2v_model = self.vec_config.get('word2vec_model', 'en_core_web_sm')
            self.text_vectorizer = Word2VecVectorizer(model_name=w2v_model)
        else:
            raise ValueError(f"Unknown vectorizer model: {self.model_type}")
        
        # Initialize time encoder
        d_max = self.vec_config.get('time_dmax', 100000)
        s_max = self.vec_config.get('time_smax', 86400)
        self.time_encoder = TimeEncoder(d_max=d_max, s_max=s_max)
        
        # Compute total embedding dimension
        self.text_dim = self.text_vectorizer.get_embedding_dim()
        self.time_dim = 2 if self.include_time else 0
        self.total_dim = self.text_dim + self.time_dim
        
        logger.info(f"Vectorizer initialized: {self.model_type}")
        logger.info(f"Text dim: {self.text_dim}, Time dim: {self.time_dim}, Total: {self.total_dim}")
    
    def vectorize(
        self,
        text: str,
        timestamp: Optional[Union[datetime, str]] = None
    ) -> np.ndarray:
        """
        Vectorize a single text with optional timestamp.
        
        Args:
            text: Input text
            timestamp: Optional timestamp
            
        Returns:
            Combined vector
        """
        # Get text embedding
        text_vec = self.text_vectorizer.encode(text)
        
        # Add time vector if enabled
        if self.include_time and timestamp is not None:
            time_vec = self.time_encoder.encode(timestamp)
            return np.concatenate([text_vec, time_vec])
        
        return text_vec
    
    def vectorize_batch(
        self,
        texts: List[str],
        timestamps: Optional[List[Union[datetime, str]]] = None,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Vectorize a batch of texts with optional timestamps.
        
        Args:
            texts: List of input texts
            timestamps: Optional list of timestamps
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of shape (n, total_dim)
        """
        # Get text embeddings
        text_vecs = self.text_vectorizer.encode_batch(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        
        # Add time vectors if enabled
        if self.include_time and timestamps is not None:
            time_vecs = self.time_encoder.encode_batch(timestamps)
            return np.concatenate([text_vecs, time_vecs], axis=1)
        
        return text_vecs
    
    def get_embedding_dim(self) -> int:
        """Get the total embedding dimension."""
        return self.total_dim
    
    def get_text_dim(self) -> int:
        """Get the text embedding dimension."""
        return self.text_dim
    
    def get_time_dim(self) -> int:
        """Get the time embedding dimension."""
        return self.time_dim


if __name__ == "__main__":
    # Test the vectorizer
    import yaml
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nTesting Time Encoder:")
    print("="*60)
    
    time_encoder = TimeEncoder()
    test_times = [
        datetime(2024, 1, 15, 10, 30, 0),
        datetime(2024, 6, 20, 18, 45, 30),
        "2024-03-10T14:20:00"
    ]
    
    for t in test_times:
        vec = time_encoder.encode(t)
        print(f"{t} -> {vec}")
    
    print("\nTesting SBERT Vectorizer:")
    print("="*60)
    
    try:
        sbert = SBERTVectorizer()
        test_texts = [
            "This is a test sentence about social events.",
            "Breaking news: Major announcement today!",
            "User posted about local weather conditions."
        ]
        
        for text in test_texts:
            vec = sbert.encode(text)
            print(f"'{text[:40]}...' -> shape={vec.shape}, mean={vec.mean():.4f}")
        
        # Batch encoding
        batch_vecs = sbert.encode_batch(test_texts)
        print(f"\nBatch encoding: shape={batch_vecs.shape}")
        
    except ImportError as e:
        print(f"SBERT not available: {e}")
    
    print("\nTesting Full Vectorizer:")
    print("="*60)
    
    try:
        vectorizer = Vectorizer(config)
        
        # Test single vectorization
        vec = vectorizer.vectorize(
            "Test message about an event",
            datetime(2024, 5, 15, 12, 0, 0)
        )
        print(f"Single vectorization: shape={vec.shape}")
        print(f"Text dim: {vectorizer.get_text_dim()}, Time dim: {vectorizer.get_time_dim()}")
        
        # Test batch vectorization
        texts = ["Message 1", "Message 2", "Message 3"]
        timestamps = [
            datetime(2024, 1, 1, 10, 0, 0),
            datetime(2024, 1, 2, 14, 30, 0),
            datetime(2024, 1, 3, 18, 45, 0)
        ]
        
        batch_vecs = vectorizer.vectorize_batch(texts, timestamps)
        print(f"Batch vectorization: shape={batch_vecs.shape}")
        
    except Exception as e:
        print(f"Vectorizer test failed: {e}")
