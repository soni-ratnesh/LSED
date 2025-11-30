"""
Hyperbolic Encoder Module for LSED
Implements hyperbolic space embeddings using Poincaré Ball and Hyperboloid models.

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

Key concepts:
1. Hyperbolic space better captures hierarchical structures in natural language
2. Poincaré Ball model performs better than Hyperboloid model
3. Uses HGCN-style encoder (Chami et al., 2019)

Mathematical definitions:
- Poincaré Ball model: exp^c_o(a) = tanh(√c||a||) * a / (√c||a||)
- Hyperboloid model: exp^c_o(x) = cosh(||x||/√c)y' + √c·sinh(||x||/√c) * x/||x||
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# HYPERBOLIC MATH OPERATIONS
# =============================================================================

class HyperbolicMath:
    """Mathematical operations for hyperbolic spaces."""
    
    @staticmethod
    def arctanh(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable arctanh."""
        x = torch.clamp(x, min=-1 + 1e-7, max=1 - 1e-7)
        return 0.5 * torch.log((1 + x) / (1 - x))
    
    @staticmethod
    def artanh(x: torch.Tensor) -> torch.Tensor:
        """Alias for arctanh."""
        return HyperbolicMath.arctanh(x)


# =============================================================================
# POINCARÉ BALL MODEL
# =============================================================================

class PoincareBall:
    """
    Poincaré Ball model for hyperbolic embeddings.
    
    The Poincaré Ball is a model of hyperbolic space where points are
    represented as vectors inside the unit ball.
    
    Key operations:
    - exp_map: Map from tangent space to hyperbolic space
    - log_map: Map from hyperbolic space to tangent space
    - distance: Compute geodesic distance in hyperbolic space
    """
    
    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        """
        Initialize Poincaré Ball model.
        
        Args:
            c: Curvature (positive for hyperbolic)
            eps: Small epsilon for numerical stability
        """
        self.c = c
        self.eps = eps
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at point x with tangent vector v.
        
        For origin (x=0):
        exp^c_o(v) = tanh(√c||v||) * v / (√c||v||)
        
        Args:
            x: Base point (origin if mapping from Euclidean)
            v: Tangent vector
            
        Returns:
            Point in Poincaré Ball
        """
        sqrt_c = np.sqrt(self.c)
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=self.eps)
        
        # exp_o^c(v) = tanh(√c||v||) * v / (√c||v||)
        result = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        
        return result
    
    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at origin (Euclidean to Poincaré).
        
        exp^c_o(v) = tanh(√c||v||) * v / (√c||v||)
        
        Args:
            v: Tangent vector at origin (Euclidean vector)
            
        Returns:
            Point in Poincaré Ball
        """
        sqrt_c = np.sqrt(self.c)
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=self.eps)
        
        result = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        
        return result
    
    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at origin (Poincaré to Euclidean).
        
        log^c_o(y) = artanh(√c||y||) * y / (√c||y||)
        
        Args:
            y: Point in Poincaré Ball
            
        Returns:
            Tangent vector at origin (Euclidean vector)
        """
        sqrt_c = np.sqrt(self.c)
        y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), min=self.eps)
        
        result = HyperbolicMath.artanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)
        
        return result
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in Poincaré Ball.
        
        Args:
            x, y: Points in Poincaré Ball
            
        Returns:
            x ⊕ y
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        
        return num / torch.clamp(denom, min=self.eps)
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance in Poincaré Ball.
        
        Args:
            x, y: Points in Poincaré Ball
            
        Returns:
            Geodesic distance
        """
        sqrt_c = np.sqrt(self.c)
        diff = self.mobius_add(-x, y)
        diff_norm = torch.clamp(torch.norm(diff, dim=-1), min=self.eps)
        
        return (2 / sqrt_c) * HyperbolicMath.artanh(sqrt_c * diff_norm)
    
    def project(self, x: torch.Tensor, max_norm: float = 1 - 1e-5) -> torch.Tensor:
        """
        Project point onto Poincaré Ball (ensure ||x|| < 1).
        
        Args:
            x: Point to project
            max_norm: Maximum norm (should be < 1)
            
        Returns:
            Projected point
        """
        norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=self.eps)
        max_norm = max_norm / np.sqrt(self.c)
        
        cond = norm > max_norm
        projected = x / norm * max_norm
        
        return torch.where(cond, projected, x)


# =============================================================================
# HYPERBOLOID MODEL
# =============================================================================

class Hyperboloid:
    """
    Hyperboloid model (Lorentz model) for hyperbolic embeddings.
    
    Points lie on the upper sheet of a two-sheeted hyperboloid.
    Uses Minkowski inner product: ⟨x, y⟩_M = -x_0*y_0 + sum(x_i*y_i)
    """
    
    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        """
        Initialize Hyperboloid model.
        
        Args:
            c: Curvature
            eps: Small epsilon for numerical stability
        """
        self.c = c
        self.eps = eps
    
    def minkowski_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Minkowski inner product.
        
        ⟨x, y⟩_M = -x_0*y_0 + sum(x_i*y_i for i > 0)
        """
        # First component has negative sign
        res = -x[..., 0:1] * y[..., 0:1] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        return res.squeeze(-1)
    
    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Minkowski norm: sqrt(|⟨x, x⟩_M|)"""
        dot = self.minkowski_dot(x, x)
        return torch.sqrt(torch.clamp(torch.abs(dot), min=self.eps))
    
    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at origin (Euclidean to Hyperboloid).
        
        exp^c_o(x) = cosh(||x||/√c)y' + √c·sinh(||x||/√c) * x/||x||
        
        For origin, this simplifies to:
        - Time component: cosh(||v||/√c)
        - Space components: sinh(||v||/√c) * v/||v||
        """
        sqrt_c = np.sqrt(self.c)
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=self.eps)
        
        # Time component
        time = torch.cosh(v_norm / sqrt_c)
        
        # Space components
        space = torch.sinh(v_norm / sqrt_c) * v / v_norm
        
        return torch.cat([time, space], dim=-1)
    
    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at origin (Hyperboloid to Euclidean).
        
        Args:
            y: Point on hyperboloid (first component is time)
            
        Returns:
            Tangent vector at origin
        """
        sqrt_c = np.sqrt(self.c)
        
        # Extract components
        time = y[..., 0:1]
        space = y[..., 1:]
        
        # Compute norm in space
        space_norm = torch.clamp(torch.norm(space, dim=-1, keepdim=True), min=self.eps)
        
        # Inverse hyperbolic functions
        factor = torch.acosh(torch.clamp(time, min=1 + self.eps)) * sqrt_c
        
        return factor * space / space_norm
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance on hyperboloid.
        
        d(x, y) = (1/√c) * arcosh(-⟨x, y⟩_M)
        """
        sqrt_c = np.sqrt(self.c)
        inner = -self.minkowski_dot(x, y)
        inner = torch.clamp(inner, min=1 + self.eps)
        
        return (1 / sqrt_c) * torch.acosh(inner)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point onto hyperboloid (ensure it satisfies constraint).
        
        The constraint is: -x_0^2 + sum(x_i^2) = -1/c
        """
        # Recompute time component from space components
        space = x[..., 1:]
        space_sqnorm = torch.sum(space * space, dim=-1, keepdim=True)
        
        time = torch.sqrt(1 / self.c + space_sqnorm)
        
        return torch.cat([time, space], dim=-1)


# =============================================================================
# HYPERBOLIC NEURAL NETWORK LAYERS
# =============================================================================

class HyperbolicLinear(nn.Module):
    """
    Hyperbolic linear layer.
    
    Maps from Euclidean input to hyperbolic output:
    1. Apply linear transformation in Euclidean space
    2. Map to hyperbolic space via exponential map
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        c: float = 1.0,
        bias: bool = True,
        model: str = "poincare"
    ):
        """
        Initialize hyperbolic linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            c: Curvature
            bias: Whether to use bias
            model: "poincare" or "hyperboloid"
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        # Euclidean linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Hyperbolic model
        if model == "poincare":
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = Hyperboloid(c=c)
        
        self.model = model
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (Euclidean)
            
        Returns:
            Output tensor (Hyperbolic)
        """
        # Euclidean linear transformation
        h = self.linear(x)
        
        # Map to hyperbolic space
        h = self.manifold.exp_map_zero(h)
        
        # Project to ensure we're in the manifold
        h = self.manifold.project(h)
        
        return h


class HyperbolicMLP(nn.Module):
    """
    Hyperbolic Multi-Layer Perceptron.
    
    Architecture from paper:
    - 3 hidden layers
    - 64 hidden dimensions
    - ReLU activation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: Optional[int] = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        c: float = 1.0,
        model: str = "poincare"
    ):
        """
        Initialize Hyperbolic MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension (paper: 64)
            output_dim: Output dimension (default: same as hidden_dim)
            num_layers: Number of layers (paper: 3)
            dropout: Dropout rate
            activation: Activation function
            c: Curvature
            model: "poincare" or "hyperboloid"
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_layers = num_layers
        self.c = c
        self.model = model
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build layers (Euclidean MLP)
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.euclidean_mlp = nn.Sequential(*layers)
        
        # Hyperbolic manifold for final projection
        if model == "poincare":
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = Hyperboloid(c=c)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.euclidean_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (Euclidean)
            
        Returns:
            Output tensor (Hyperbolic)
        """
        # Pass through Euclidean MLP
        h = self.euclidean_mlp(x)
        
        # Map to hyperbolic space
        h = self.manifold.exp_map_zero(h)
        
        # Project to ensure we're in the manifold
        h = self.manifold.project(h)
        
        return h
    
    def encode_to_euclidean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode to Euclidean space (before hyperbolic mapping).
        
        Args:
            x: Input tensor
            
        Returns:
            Euclidean embeddings
        """
        return self.euclidean_mlp(x)
    
    def get_euclidean_embeddings(self, hyperbolic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert hyperbolic embeddings back to Euclidean space.
        
        Args:
            hyperbolic_embeddings: Points in hyperbolic space
            
        Returns:
            Euclidean vectors
        """
        return self.manifold.log_map_zero(hyperbolic_embeddings)


# =============================================================================
# MAIN HYPERBOLIC ENCODER
# =============================================================================

class HyperbolicEncoder(nn.Module):
    """
    Main Hyperbolic Encoder for LSED.
    
    Takes vectorized social message representations and encodes them
    into hyperbolic space.
    
    Based on HGCN encoder (Chami et al., 2019).
    """
    
    def __init__(self, config: Dict[str, Any], input_dim: int):
        """
        Initialize Hyperbolic Encoder.
        
        Args:
            config: Configuration dictionary
            input_dim: Input dimension (from vectorizer)
        """
        super().__init__()
        
        self.config = config
        self.hyp_config = config.get('hyperbolic', {})
        
        # Get settings
        self.model_type = self.hyp_config.get('model', 'poincare')
        self.c = self.hyp_config.get('curvature', 1.0)
        self.hidden_dim = self.hyp_config.get('hidden_dim', 64)
        self.num_layers = self.hyp_config.get('num_hidden_layers', 3)
        self.dropout = self.hyp_config.get('dropout', 0.1)
        self.activation = self.hyp_config.get('activation', 'relu')
        
        # Build encoder
        self.encoder = HyperbolicMLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            activation=self.activation,
            c=self.c,
            model=self.model_type
        )
        
        # Manifold for distance computations
        if self.model_type == "poincare":
            self.manifold = PoincareBall(c=self.c)
        else:
            self.manifold = Hyperboloid(c=self.c)
        
        logger.info(f"Hyperbolic Encoder initialized")
        logger.info(f"  Model: {self.model_type}")
        logger.info(f"  Curvature: {self.c}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        logger.info(f"  Num layers: {self.num_layers}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode vectors into hyperbolic space.
        
        Args:
            x: Input vectors (Euclidean)
            
        Returns:
            Hyperbolic embeddings
        """
        return self.encoder(x)
    
    def get_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance matrix in hyperbolic space.
        
        Args:
            embeddings: Hyperbolic embeddings
            
        Returns:
            Distance matrix
        """
        n = embeddings.shape[0]
        distances = torch.zeros(n, n, device=embeddings.device)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self.manifold.distance(embeddings[i], embeddings[j])
                distances[i, j] = d
                distances[j, i] = d
        
        return distances
    
    def get_embeddings_numpy(self, x: torch.Tensor) -> np.ndarray:
        """
        Get embeddings as numpy array.
        
        Args:
            x: Input vectors
            
        Returns:
            Numpy array of embeddings
        """
        with torch.no_grad():
            embeddings = self.forward(x)
            return embeddings.cpu().numpy()


if __name__ == "__main__":
    # Test hyperbolic encoder
    import yaml
    
    print("Testing Hyperbolic Encoder Components")
    print("="*60)
    
    # Test Poincaré Ball
    print("\n1. Poincaré Ball Operations:")
    poincare = PoincareBall(c=1.0)
    
    # Test exp/log maps
    v = torch.randn(5, 10)
    h = poincare.exp_map_zero(v)
    v_reconstructed = poincare.log_map_zero(h)
    
    print(f"   Input shape: {v.shape}")
    print(f"   Hyperbolic shape: {h.shape}")
    print(f"   Norms in ball: {torch.norm(h, dim=-1)}")  # Should be < 1
    print(f"   Reconstruction error: {torch.norm(v - v_reconstructed):.6f}")
    
    # Test distance
    x = poincare.exp_map_zero(torch.randn(3, 10))
    y = poincare.exp_map_zero(torch.randn(3, 10))
    dist = poincare.distance(x, y)
    print(f"   Distances: {dist}")
    
    # Test Hyperboloid
    print("\n2. Hyperboloid Operations:")
    hyperboloid = Hyperboloid(c=1.0)
    
    v = torch.randn(5, 10)
    h = hyperboloid.exp_map_zero(v)
    
    print(f"   Input shape: {v.shape}")
    print(f"   Hyperboloid shape: {h.shape}")  # Should be (5, 11) - extra time dim
    
    # Test HyperbolicMLP
    print("\n3. Hyperbolic MLP:")
    
    mlp = HyperbolicMLP(
        input_dim=384,  # SBERT dimension
        hidden_dim=64,
        num_layers=3,
        c=1.0,
        model="poincare"
    )
    
    x = torch.randn(32, 384)
    h = mlp(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {h.shape}")
    print(f"   Output norms (should be < 1): {torch.norm(h, dim=-1)[:5]}")
    
    # Test full encoder with config
    print("\n4. Full Hyperbolic Encoder:")
    
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    encoder = HyperbolicEncoder(config, input_dim=386)  # 384 + 2 (time)
    
    x = torch.randn(64, 386)
    embeddings = encoder(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Embedding shape: {embeddings.shape}")
    
    # Get numpy embeddings
    embeddings_np = encoder.get_embeddings_numpy(x)
    print(f"   Numpy embeddings shape: {embeddings_np.shape}")
    
    print("\n" + "="*60)
    print("All tests passed!")
