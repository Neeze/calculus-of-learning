import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class PredictorMLP(nn.Module):
    features: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        for feat in self.features:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.output_dim)(x)
        return x

class BlockPredictorMLP(nn.Module):
    """
    Block predictor: takes s_t and the k actions a_{t..t+k-1} (flattened)
    and outputs the whole block \\hat{s}_{t+1..t+k} in one forward pass.
    This is the "higher-order" / jumpy predictor needed to test P1b —
    it is architecturally distinct from M1, not just a different loss on M1.
    """
    features: Sequence[int]
    k: int
    d: int

    @nn.compact
    def __call__(self, state, actions):
        # actions: (..., k, d_action) -> flatten the block of actions
        actions_flat = actions.reshape(*actions.shape[:-2], self.k * actions.shape[-1])
        x = jnp.concatenate([state, actions_flat], axis=-1)
        for feat in self.features:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.k * self.d)(x)
        x = x.reshape(*x.shape[:-1], self.k, self.d)
        return x

def create_m1_model(d: int = 8, hidden_dims: Sequence[int] = (256, 384)):
    r"""
    Creates One-Step Predictor (M1)
    Outputs: shape (d,) representing \hat{s}_{t+1}
    """
    return PredictorMLP(features=hidden_dims, output_dim=d)

def create_mk_model(d: int = 8, k: int = 4, hidden_dims: Sequence[int] = (256, 384)):
    r"""
    Creates Block Predictor (M2, M3): analogue of a higher-order (RK-like)
    integrator step. Predicts the whole block \\hat{s}_{t+1..t+k} from
    (s_t, a_{t..t+k-1}) in a single forward pass, so rollout jumps k steps
    at a time instead of composing k one-step predictions.
    """
    return BlockPredictorMLP(features=hidden_dims, k=k, d=d)
