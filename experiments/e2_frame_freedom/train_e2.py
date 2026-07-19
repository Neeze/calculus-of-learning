import os
import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pickle
from flax import linen as nn
from flax.training import train_state

from src.envs.dmc_ood import make_dmc_env
from src.models.regularizers import sigreg_loss, vicreg_loss, reconstruction_loss

class Encoder(nn.Module):
    hidden_dim: int = 256
    latent_dim: int = 32
    
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        z = nn.Dense(self.latent_dim)(x)
        return z

class Decoder(nn.Module):
    hidden_dim: int = 256
    obs_dim: int
    
    @nn.compact
    def __call__(self, z):
        x = nn.relu(nn.Dense(self.hidden_dim)(z))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x_rec = nn.Dense(self.obs_dim)(x)
        return x_rec

class Dynamics(nn.Module):
    hidden_dim: int = 256
    latent_dim: int = 32
    
    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        z_next = nn.Dense(self.latent_dim)(x)
        return z_next

class WorldModel(nn.Module):
    obs_dim: int
    latent_dim: int = 32
    
    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder(obs_dim=self.obs_dim)
        self.dynamics = Dynamics(latent_dim=self.latent_dim)
        
    def __call__(self, obs, action):
        z = self.encoder(obs)
        z_next_pred = self.dynamics(z, action)
        obs_rec = self.decoder(z)
        return z, z_next_pred, obs_rec

def collect_random_dataset(env, num_samples: int = 10000):
    """Collects transitions using a random policy"""
    states, actions, next_states = [], [], []
    s = env.reset()
    # Assume 1D action space for simplicity of DMC toy
    # Let's get action spec
    from dm_env import specs
    action_spec = env.env.action_spec()
    action_dim = action_spec.shape[0]
    
    for _ in range(num_samples):
        # sample random action between min and max
        a = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        s_next, _, done, _ = env.step(a)
        
        states.append(s)
        actions.append(a)
        next_states.append(s_next)
        
        s = s_next
        if done:
            s = env.reset()
            
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'next_states': np.array(next_states)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='walker-walk')
    parser.add_argument('--reg', type=str, default='jepa') # full_rec, light_rec, vicreg, light_vicreg, jepa
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Generate data
    print(f"Collecting data for {args.env} (seed={args.seed})...")
    env = make_dmc_env(*args.env.split("-"), mode="train", seed=args.seed)
    train_data = collect_random_dataset(env, num_samples=20000)
    
    obs_dim = train_data['states'].shape[1]
    action_dim = train_data['actions'].shape[1]
    
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    
    model = WorldModel(obs_dim=obs_dim)
    params = model.init(init_rng, jnp.ones((1, obs_dim)), jnp.ones((1, action_dim)))['params']
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    @jax.jit
    def train_step(state, batch, rng_key):
        def loss_fn(params):
            z, z_next_pred, obs_rec = state.apply_fn({'params': params}, batch['states'], batch['actions'])
            z_next_target = state.apply_fn({'params': params}, batch['next_states'], batch['actions'], method=lambda m, s, a: m.encoder(s))
            
            # Latent prediction loss
            pred_loss = jnp.mean((z_next_pred - jax.lax.stop_gradient(z_next_target)) ** 2)
            
            # Regularizer loss
            reg_loss = 0.0
            if args.reg == 'full_rec':
                reg_loss = 1.0 * reconstruction_loss(obs_rec, batch['states'])
            elif args.reg == 'light_rec':
                reg_loss = 0.1 * reconstruction_loss(obs_rec, batch['states'])
            elif args.reg == 'vicreg':
                reg_loss = vicreg_loss(z, var_weight=25.0, cov_weight=1.0)
            elif args.reg == 'light_vicreg':
                reg_loss = vicreg_loss(z, var_weight=6.25, cov_weight=0.25)
            elif args.reg == 'jepa':
                reg_loss = 1.0 * sigreg_loss(z, rng_key)
                
            loss = pred_loss + reg_loss
            return loss, (pred_loss, reg_loss)
            
        (loss, (pred_loss, reg_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, pred_loss, reg_loss

    batch_size = 256
    num_epochs = 100
    num_batches = len(train_data['states']) // batch_size
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(len(train_data['states']))
        for i in range(num_batches):
            rng, step_rng = jax.random.split(rng)
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            batch = {k: v[batch_idx] for k, v in train_data.items()}
            state, loss, pred_loss, reg_loss = train_step(state, batch, step_rng)
            
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}: Total={loss:.4f} Pred={pred_loss:.4f} Reg={reg_loss:.4f}")
            
    os.makedirs("outputs/checkpoints/e2", exist_ok=True)
    with open(f"outputs/checkpoints/e2/model_{args.env}_{args.reg}_seed{args.seed}.pkl", "wb") as f:
        pickle.dump(state.params, f)
    
    print(f"Model saved for {args.reg} seed {args.seed}.")

if __name__ == "__main__":
    main()
