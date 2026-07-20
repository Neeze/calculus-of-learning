"""E2 training (spec §3.2-3.3, §7).

One run = (env, regularizer config, seed). All 5 configs of a seed train on the
SAME cached dataset (collect_e2.py) so the objective is the only difference.
Train/val split is by trajectory; the checkpoint is selected by validation
one-step latent prediction loss, not the final epoch.
"""
import os
import json
import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pickle
from flax import linen as nn
from flax.training import train_state

from src.models.regularizers import sigreg_loss, vicreg_loss, reconstruction_loss
from experiments.e2_frame_freedom.collect_e2 import dataset_path, load_dataset, flatten_transitions

REGULARIZERS = ["full_rec", "light_rec", "vicreg", "light_vicreg", "sigreg"]
CKPT_DIR = "outputs/checkpoints/e2"


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
    obs_dim: int = 24

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


class RewardPredictor(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        r = nn.Dense(1)(x)
        return r.squeeze(-1)


class WorldModel(nn.Module):
    obs_dim: int
    latent_dim: int = 32

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder(obs_dim=self.obs_dim)
        self.dynamics = Dynamics(latent_dim=self.latent_dim)
        self.reward_predictor = RewardPredictor()

    def __call__(self, obs, action):
        z = self.encoder(obs)
        z_next_pred = self.dynamics(z, action)
        obs_rec = self.decoder(z)
        reward_pred = self.reward_predictor(z, action)
        return z, z_next_pred, obs_rec, reward_pred

    def encode(self, obs):
        return self.encoder(obs)

    def predict(self, z, action):
        return self.dynamics(z, action)

    def predict_reward(self, z, action):
        return self.reward_predictor(z, action)


def reg_loss_fn(reg: str, z, obs_rec, obs, rng_key):
    if reg == "full_rec":
        return 1.0 * reconstruction_loss(obs_rec, obs)
    if reg == "light_rec":
        return 0.1 * reconstruction_loss(obs_rec, obs)
    if reg == "vicreg":
        return vicreg_loss(z, var_weight=25.0, cov_weight=1.0)
    if reg == "light_vicreg":
        return vicreg_loss(z, var_weight=6.25, cov_weight=0.25)
    if reg in ("sigreg", "jepa"):
        return 1.0 * sigreg_loss(z, rng_key)
    raise ValueError(f"Unknown regularizer: {reg}")


def split_train_val(data, val_frac=0.1, seed=0):
    """Split by trajectory (spec §3.3): no leakage within a trajectory."""
    n_traj = data["obs"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_traj)
    n_val = max(1, int(n_traj * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    take = lambda idx: {k: v[idx] for k, v in data.items()}
    return take(train_idx), take(val_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker-walk")
    parser.add_argument("--reg", type=str, default="sigreg", choices=REGULARIZERS + ["jepa"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    args = parser.parse_args()

    path = dataset_path(args.env, "train", seed=args.seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run collect_e2.py first — datasets must be "
            f"cached and shared across configs (spec §3.3)."
        )
    raw = load_dataset(path)
    train_raw, val_raw = split_train_val(raw, val_frac=0.1, seed=args.seed)
    train_data = flatten_transitions(train_raw)
    val_data = flatten_transitions(val_raw)

    obs_dim = train_data["states"].shape[1]
    action_dim = train_data["actions"].shape[1]
    print(f"Env {args.env} reg={args.reg} seed={args.seed} | obs_dim={obs_dim} "
          f"action_dim={action_dim} | train={len(train_data['states'])} val={len(val_data['states'])}")

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    model = WorldModel(obs_dim=obs_dim, latent_dim=args.latent_dim)
    params = model.init(init_rng, jnp.ones((1, obs_dim)), jnp.ones((1, action_dim)))["params"]
    tx = optax.adam(args.lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch, rng_key):
        def loss_fn(params):
            z, z_next_pred, obs_rec, reward_pred = state.apply_fn(
                {"params": params}, batch["states"], batch["actions"])
            z_next_target = state.apply_fn(
                {"params": params}, batch["next_states"], method=WorldModel.encode)

            pred_loss = jnp.mean((z_next_pred - jax.lax.stop_gradient(z_next_target)) ** 2)
            reward_loss = jnp.mean((reward_pred - batch["rewards"]) ** 2)
            reg_loss = reg_loss_fn(args.reg, z, obs_rec, batch["states"], rng_key)

            loss = pred_loss + reward_loss + reg_loss
            return loss, (pred_loss, reward_loss, reg_loss)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    @jax.jit
    def val_pred_loss(params, batch):
        z = model.apply({"params": params}, batch["states"], method=WorldModel.encode)
        z_next_pred = model.apply({"params": params}, z, batch["actions"], method=WorldModel.predict)
        z_next_target = model.apply({"params": params}, batch["next_states"], method=WorldModel.encode)
        # Normalize by latent variance so the selection criterion is scale-free:
        # otherwise a shrinking latent (partial collapse) fakes a low val loss.
        var = jnp.mean(jnp.var(z_next_target, axis=0)) + 1e-8
        return jnp.mean((z_next_pred - z_next_target) ** 2) / var

    val_batch = {k: jnp.array(v) for k, v in val_data.items()}
    np_rng = np.random.default_rng(args.seed)
    n = len(train_data["states"])
    num_batches = n // args.batch_size

    best_val, best_params, best_epoch = np.inf, None, -1
    for epoch in range(args.epochs):
        indices = np_rng.permutation(n)
        for i in range(num_batches):
            rng, step_rng = jax.random.split(rng)
            idx = indices[i * args.batch_size:(i + 1) * args.batch_size]
            batch = {k: jnp.array(v[idx]) for k, v in train_data.items()}
            state, loss, (pred_loss, reward_loss, reg_loss) = train_step(state, batch, step_rng)

        v = float(val_pred_loss(state.params, val_batch))
        if v < best_val:
            best_val, best_epoch = v, epoch
            best_params = jax.tree_util.tree_map(np.array, state.params)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch}: Total={loss:.4f} Pred={pred_loss:.4f} "
                  f"Rew={reward_loss:.4f} Reg={reg_loss:.4f} | ValPred(norm)={v:.4f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"{args.env}_{args.reg}_seed{args.seed}"
    with open(os.path.join(CKPT_DIR, f"model_{tag}.pkl"), "wb") as f:
        pickle.dump(best_params, f)
    run_config = {
        **vars(args),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "best_val_pred_loss": best_val,
        "best_epoch": best_epoch,
        "dataset": path,
    }
    with open(os.path.join(CKPT_DIR, f"config_{tag}.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Saved best checkpoint (epoch {best_epoch}, val={best_val:.4f}) for {tag}.")


if __name__ == "__main__":
    main()
