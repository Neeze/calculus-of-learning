import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class PredictorMLP(nn.Module):
    features: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, state, action):
        # Concatenate state and action
        x = jnp.concatenate([state, action], axis=-1)
        for feat in self.features:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.output_dim)(x)
        return x

def create_m1_model(d: int = 8, hidden_dims: Sequence[int] = (256, 384)):
    """
    Creates One-Step Predictor (M1)
    Outputs: shape (d,) representing \hat{s}_{t+1}
    """
    return PredictorMLP(features=hidden_dims, output_dim=d)

def create_mk_model(d: int = 8, k: int = 4, hidden_dims: Sequence[int] = (256, 384)):
    """
    Creates Multi-Step Predictor (M2, M3)
    Outputs: shape (k * d,) which will be reshaped to (k, d) representing \hat{s}_{t+1..t+k}
    """
    return PredictorMLP(features=hidden_dims, output_dim=k * d)
