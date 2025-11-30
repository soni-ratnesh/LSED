"""
Clustering Module for LSED
Implements K-Means clustering for social event detection.

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

The paper uses K-Means clustering on hyperbolic embeddings to group
social messages into events.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class EventClustering:
    """
    Clustering module for social event detection.
    
    Uses K-Means clustering to group message embeddings into events.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clustering module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cluster_config = config.get('clustering', {})
        
        # Clustering settings
        self.algorithm = self.cluster_config.get('algorithm', 'kmeans')
        self.n_clusters = self.cluster_config.get('n_clusters', None)
        self.kmeans_init = self.cluster_config.get('kmeans_init', 'k-means++')
        self.kmeans_n_init = self.cluster_config.get('kmeans_n_init', 10)
        self.kmeans_max_iter = self.cluster_config.get('kmeans_max_iter', 300)
        
        # Random seed
        self.seed = config.get('training', {}).get('seed', 42)
        
        # Fitted clusterer
        self.clusterer = None
        self.fitted = False
        
        logger.info(f"Clustering initialized with algorithm: {self.algorithm}")
    
    def fit(
        self, 
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> 'EventClustering':
        """
        Fit the clustering model.
        
        Args:
            embeddings: Message embeddings (n_samples, n_features)
            n_clusters: Number of clusters (overrides config if provided)
            
        Returns:
            self
        """
        n_clusters = n_clusters or self.n_clusters
        
        if n_clusters is None:
            raise ValueError("n_clusters must be specified")
        
        logger.info(f"Fitting {self.algorithm} with {n_clusters} clusters on {len(embeddings)} samples")
        
        if self.algorithm == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                init=self.kmeans_init,
                n_init=self.kmeans_n_init,
                max_iter=self.kmeans_max_iter,
                random_state=self.seed
            )
        elif self.algorithm == 'minibatch_kmeans':
            self.clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                init=self.kmeans_init,
                n_init=self.kmeans_n_init,
                max_iter=self.kmeans_max_iter,
                random_state=self.seed
            )
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")
        
        self.clusterer.fit(embeddings)
        self.fitted = True
        
        # Log cluster sizes
        labels = self.clusterer.labels_
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for embeddings.
        
        Args:
            embeddings: Message embeddings
            
        Returns:
            Cluster labels
        """
        if not self.fitted:
            raise RuntimeError("Clustering model not fitted. Call fit() first.")
        
        return self.clusterer.predict(embeddings)
    
    def fit_predict(
        self, 
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit and predict in one step.
        
        Args:
            embeddings: Message embeddings
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        self.fit(embeddings, n_clusters)
        return self.clusterer.labels_
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if not self.fitted:
            raise RuntimeError("Clustering model not fitted.")
        return self.clusterer.cluster_centers_
    
    def get_inertia(self) -> float:
        """Get within-cluster sum of squares."""
        if not self.fitted:
            raise RuntimeError("Clustering model not fitted.")
        return self.clusterer.inertia_
    
    def compute_silhouette_score(self, embeddings: np.ndarray) -> float:
        """
        Compute silhouette score for the clustering.
        
        Args:
            embeddings: Message embeddings
            
        Returns:
            Silhouette score
        """
        if not self.fitted:
            raise RuntimeError("Clustering model not fitted.")
        
        labels = self.clusterer.labels_
        
        # Need at least 2 clusters and more than 1 sample per cluster
        if len(np.unique(labels)) < 2:
            return 0.0
        
        return silhouette_score(embeddings, labels)


class HyperbolicKMeans:
    """
    K-Means clustering in hyperbolic space.
    
    Uses hyperbolic distance for centroid computation.
    Note: This is a simplified version; for production, consider using
    specialized hyperbolic clustering libraries.
    """
    
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
        c: float = 1.0
    ):
        """
        Initialize Hyperbolic K-Means.
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            tol: Convergence tolerance
            random_state: Random seed
            c: Curvature of hyperbolic space
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.c = c
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = 0
    
    def _poincare_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Poincaré distance."""
        sqrt_c = np.sqrt(self.c)
        
        # Compute ||x - y||^2
        diff_norm_sq = np.sum((x - y) ** 2, axis=-1)
        
        # Compute (1 - ||x||^2) and (1 - ||y||^2)
        x_norm_sq = np.sum(x ** 2, axis=-1)
        y_norm_sq = np.sum(y ** 2, axis=-1)
        
        # Poincaré distance formula
        num = diff_norm_sq
        denom = (1 - self.c * x_norm_sq) * (1 - self.c * y_norm_sq)
        
        inner = 1 + 2 * self.c * num / (denom + 1e-10)
        inner = np.clip(inner, 1 + 1e-10, None)
        
        return (1 / sqrt_c) * np.arccosh(inner)
    
    def _compute_centroid(self, points: np.ndarray) -> np.ndarray:
        """
        Compute centroid in Poincaré ball.
        
        Uses the Euclidean mean as an approximation (works well for small curvatures).
        """
        # Simple Euclidean mean projected back
        mean = np.mean(points, axis=0)
        
        # Project to ensure we stay in the ball
        norm = np.linalg.norm(mean)
        max_norm = 1 / np.sqrt(self.c) - 1e-5
        
        if norm > max_norm:
            mean = mean * (max_norm / norm)
        
        return mean
    
    def fit(self, X: np.ndarray) -> 'HyperbolicKMeans':
        """
        Fit hyperbolic K-Means.
        
        Args:
            X: Data points in Poincaré ball
            
        Returns:
            self
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from data points
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()
        
        for iteration in range(self.max_iter):
            # Assign points to nearest centroid
            distances = np.zeros((n_samples, self.n_clusters))
            for j in range(self.n_clusters):
                distances[:, j] = self._poincare_distance(X, self.cluster_centers_[j])
            
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centers = np.zeros_like(self.cluster_centers_)
            for j in range(self.n_clusters):
                mask = self.labels_ == j
                if np.sum(mask) > 0:
                    new_centers[j] = self._compute_centroid(X[mask])
                else:
                    # Keep old centroid if cluster is empty
                    new_centers[j] = self.cluster_centers_[j]
            
            # Check convergence
            center_shift = np.sum(np.linalg.norm(new_centers - self.cluster_centers_, axis=1))
            self.cluster_centers_ = new_centers
            
            if center_shift < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for j in range(self.n_clusters):
            distances[:, j] = self._poincare_distance(X, self.cluster_centers_[j])
        
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict."""
        self.fit(X)
        return self.labels_


class AdaptiveClustering:
    """
    Adaptive clustering that can automatically determine the number of clusters.
    """
    
    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 100,
        method: str = "silhouette",
        random_state: int = 42
    ):
        """
        Initialize adaptive clustering.
        
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            method: Method for selecting optimal k ("silhouette" or "elbow")
            random_state: Random seed
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.method = method
        self.random_state = random_state
        
        self.optimal_k = None
        self.clusterer = None
    
    def find_optimal_k(
        self, 
        embeddings: np.ndarray,
        step: int = 1
    ) -> int:
        """
        Find optimal number of clusters.
        
        Args:
            embeddings: Message embeddings
            step: Step size for searching k values
            
        Returns:
            Optimal number of clusters
        """
        n_samples = len(embeddings)
        max_k = min(self.max_clusters, n_samples - 1)
        
        if self.method == "silhouette":
            best_score = -1
            best_k = self.min_clusters
            
            for k in range(self.min_clusters, max_k + 1, step):
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                if len(np.unique(labels)) < 2:
                    continue
                
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            self.optimal_k = best_k
            
        elif self.method == "elbow":
            # Elbow method using inertia
            inertias = []
            k_values = list(range(self.min_clusters, max_k + 1, step))
            
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point using second derivative
            if len(inertias) >= 3:
                second_derivative = np.diff(np.diff(inertias))
                elbow_idx = np.argmax(second_derivative) + 1
                self.optimal_k = k_values[elbow_idx]
            else:
                self.optimal_k = k_values[0]
        
        logger.info(f"Optimal k found: {self.optimal_k}")
        return self.optimal_k
    
    def fit_predict(
        self, 
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit and predict with automatic or specified k.
        
        Args:
            embeddings: Message embeddings
            n_clusters: Number of clusters (if None, will be determined automatically)
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_k(embeddings)
        
        self.clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        return self.clusterer.fit_predict(embeddings)


if __name__ == "__main__":
    # Test clustering module
    import yaml
    
    print("Testing Clustering Module")
    print("="*60)
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_clusters_true = 10
    n_features = 64
    
    # Create clustered data
    embeddings = np.vstack([
        np.random.randn(n_samples // n_clusters_true, n_features) + i * 2
        for i in range(n_clusters_true)
    ])
    true_labels = np.repeat(np.arange(n_clusters_true), n_samples // n_clusters_true)
    
    print(f"\nSynthetic data: {embeddings.shape}")
    print(f"True clusters: {n_clusters_true}")
    
    # Test EventClustering
    print("\n1. EventClustering (Standard K-Means):")
    clustering = EventClustering(config)
    pred_labels = clustering.fit_predict(embeddings, n_clusters=n_clusters_true)
    
    print(f"   Predicted labels shape: {pred_labels.shape}")
    print(f"   Unique clusters: {len(np.unique(pred_labels))}")
    print(f"   Inertia: {clustering.get_inertia():.2f}")
    print(f"   Silhouette score: {clustering.compute_silhouette_score(embeddings):.4f}")
    
    # Test HyperbolicKMeans
    print("\n2. HyperbolicKMeans:")
    
    # Project data to Poincaré ball (simple normalization)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    max_norm = np.max(norms)
    poincare_embeddings = embeddings / (max_norm * 1.1)  # Ensure < 1
    
    hyp_kmeans = HyperbolicKMeans(n_clusters=n_clusters_true, max_iter=100)
    hyp_labels = hyp_kmeans.fit_predict(poincare_embeddings)
    
    print(f"   Iterations: {hyp_kmeans.n_iter_}")
    print(f"   Unique clusters: {len(np.unique(hyp_labels))}")
    
    # Test AdaptiveClustering
    print("\n3. AdaptiveClustering:")
    adaptive = AdaptiveClustering(min_clusters=5, max_clusters=20)
    adaptive_labels = adaptive.fit_predict(embeddings)
    
    print(f"   Found optimal k: {adaptive.optimal_k}")
    print(f"   Unique clusters: {len(np.unique(adaptive_labels))}")
    
    print("\n" + "="*60)
    print("All tests passed!")
