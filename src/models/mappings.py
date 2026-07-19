import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

class IdentityMap:
    def __init__(self):
        pass
    def fit(self, ZA, ZB):
        pass
    def __call__(self, z):
        return z

class OrthogonalProcrustesMap:
    def __init__(self):
        self.Q = None
        
    def fit(self, ZA, ZB):
        # Find orthogonal Q that minimizes ||ZB - ZA @ Q||_F
        # Equivalently, max Tr(Q^T ZA^T ZB)
        M = np.dot(ZA.T, ZB)
        U, _, Vt = np.linalg.svd(M)
        self.Q = np.dot(U, Vt)
        
    def __call__(self, z):
        return np.dot(z, self.Q)

class AffineMap:
    def __init__(self):
        self.W = None
        self.b = None
        
    def fit(self, ZA, ZB):
        # Fit least squares ZA @ W + b \approx ZB
        # Add bias trick
        ZA_bias = np.concatenate([ZA, np.ones((ZA.shape[0], 1))], axis=1)
        W_bias, _, _, _ = np.linalg.lstsq(ZA_bias, ZB, rcond=None)
        self.W = W_bias[:-1, :]
        self.b = W_bias[-1, :]
        
    def __call__(self, z):
        return np.dot(z, self.W) + self.b

class RelativeRepMap:
    def __init__(self):
        self.anchors_A = None
        self.W = None
        self.b = None
        
    def fit(self, ZA, ZB):
        # We store the anchors to compute relative representations
        # To avoid large matrices, select min(500, N) anchors randomly
        num_anchors = min(500, ZA.shape[0])
        idx = np.random.choice(ZA.shape[0], num_anchors, replace=False)
        self.anchors_A = ZA[idx]
        
        # Compute rel_A for the whole training set
        rel_A_train = self._cosine_similarity(ZA, self.anchors_A)
        
        # Learn linear map from rel_A_train to ZB
        rel_A_bias = np.concatenate([rel_A_train, np.ones((rel_A_train.shape[0], 1))], axis=1)
        W_bias, _, _, _ = np.linalg.lstsq(rel_A_bias, ZB, rcond=None)
        self.W = W_bias[:-1, :]
        self.b = W_bias[-1, :]
        
    def _cosine_similarity(self, z, anchors):
        # z: (N, D), anchors: (M, D)
        # Returns: (N, M)
        z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)
        a_norm = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-6)
        return np.dot(z_norm, a_norm.T)
        
    def __call__(self, z):
        rel_A = self._cosine_similarity(z, self.anchors_A)
        return np.dot(rel_A, self.W) + self.b

# We will combine Relative Rep into a simple similarity-based mapping.
class SemanticAlignmentMap:
    def __init__(self):
        self.cca = None
        
    def fit(self, ZA, ZB):
        # Semantic alignment via Canonical Correlation Analysis (CCA)
        from sklearn.cross_decomposition import CCA
        # Limit components to the feature dimension (max 20 for stability)
        n_comp = min(ZA.shape[1], 20)
        self.cca = CCA(n_components=n_comp)
        self.cca.fit(ZA, ZB)
        
    def __call__(self, z):
        # CCA returns the projection. We use predict to map to ZB space
        return self.cca.predict(z)

class MLPMapper(nn.Module):
    output_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.Dense(self.output_dim)(x)
        return x

@jax.jit
def _mlp_train_step(state, batch_A, batch_B):
    def loss_fn(p):
        pred_B = state.apply_fn({'params': p}, batch_A)
        return jnp.mean((pred_B - batch_B) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

class MLPMap:
    def __init__(self, output_dim: int):
        self.output_dim = output_dim
        self.model = MLPMapper(output_dim=output_dim)
        self.params = None
        
    def fit(self, ZA, ZB, epochs=200):
        rng = jax.random.PRNGKey(0)
        params = self.model.init(rng, jnp.ones((1, ZA.shape[1])))['params']
        tx = optax.adam(1e-3)
        state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)
        
        ZA_jnp = jnp.array(ZA)
        ZB_jnp = jnp.array(ZB)
        
        batch_size = 128
        num_batches = len(ZA) // batch_size
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(ZA))
            for i in range(num_batches):
                idx = indices[i*batch_size:(i+1)*batch_size]
                state, loss = _mlp_train_step(state, ZA_jnp[idx], ZB_jnp[idx])
                
        self.params = state.params
        
    def __call__(self, z):
        return np.array(self.model.apply({'params': self.params}, jnp.array(z)))
