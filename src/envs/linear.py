import numpy as np
from typing import Tuple

class LinearSystemEnv:
    def __init__(
        self,
        d: int = 8,
        target_L: float = 1.05,
        noise_std: float = 0.01,
        seed: int = 42,
        spectrum_low_mult: float = 0.9,
    ):
        self.d = d
        self.target_L = target_L
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        A_init = self.rng.normal(0, 1, (d, d))

        U, S, Vt = np.linalg.svd(A_init)

        # Spectrum narrow around target_L (default [0.9*L, L]) so that
        # ||A||_2 = L is the dominant quantity governing E(h), rather than
        # a wide spectrum where most modes contract much faster than L.
        S = np.linspace(spectrum_low_mult * target_L, target_L, d)
        self.A = U @ np.diag(S) @ Vt

        self.B = self.rng.normal(0, 0.5, (d, d))

        self.state = np.zeros(d)

    def reset(self) -> np.ndarray:
        self.state = self.rng.normal(0, 1, self.d)
        return self.state.copy()

    def step(self, action: np.ndarray, noise_std: float = None) -> Tuple[np.ndarray, float, bool, dict]:
        eff_noise_std = self.noise_std if noise_std is None else noise_std
        noise = self.rng.normal(0, eff_noise_std, self.d) if eff_noise_std > 0 else np.zeros(self.d)
        self.state = self.A @ self.state + self.B @ action + noise

        return self.state.copy(), 0.0, False, {}

def generate_linear_dataset(
    target_L: float = 1.05,
    d: int = 8,
    num_transitions: int = 50000,
    traj_length: int = 100,
    seed: int = 42,
    noise_std: float = 0.01,
    test_noise_std: float = 1e-3,
    spectrum_low_mult: float = 0.9,
    max_growth_factor: float = 50.0,
):
    """
    Generates a dataset of trajectories for the linear system.
    Policy: 50% random Gaussian, 50% oscillating (sine waves).
    Returns (train_data, val_data, test_data, A, B) where each *_data is a
    dict of 'states', 'actions', 'next_states'.

    Train/val trajectories use `noise_std`. Test trajectories use a much
    smaller `test_noise_std` (default 1e-3, per E1 spec §5.3/§3.1) instead of
    the train-time noise_std, so E(h) on test measures model error rather
    than being dominated by the noise floor of a single stochastic
    realization. `test_noise_std=0.0` gives a fully deterministic test
    rollout (oracle error exactly 0); a small nonzero value keeps a
    well-defined, non-degenerate noise floor for h*(tau) (see eval_toy.py).

    For expansive systems (target_L > 1), a fixed `traj_length` (e.g. 100)
    lets the state grow ~target_L**traj_length, which overflows to huge
    magnitudes (e.g. 1.2**100 ~ 8e7) well before training even starts,
    producing exploding MSE/gradients for every model, not just the block
    predictors. `traj_length` is therefore capped so state magnitude grows
    by at most `max_growth_factor` over a trajectory (spec §5.3.4 applies
    this idea at eval time; it must also apply at data-generation time).
    """
    if target_L > 1.0:
        growth_cap = int(np.log(max_growth_factor) / np.log(target_L))
        traj_length = max(10, min(traj_length, growth_cap))

    env = LinearSystemEnv(
        d=d, target_L=target_L, noise_std=noise_std, seed=seed,
        spectrum_low_mult=spectrum_low_mult,
    )
    num_trajs = num_transitions // traj_length

    n_train = int(0.8 * num_trajs)
    n_val = int(0.1 * num_trajs)

    states, actions, next_states = [], [], []

    rng = np.random.default_rng(seed)

    for i in range(num_trajs):
        s = env.reset()
        is_oscillating = (i % 2 == 0)
        is_test = i >= n_train + n_val
        step_noise_std = test_noise_std if is_test else noise_std

        freq = rng.uniform(0.1, 1.0, d)
        phase = rng.uniform(0, 2*np.pi, d)

        for t in range(traj_length):
            if is_oscillating:
                a = np.sin(freq * t + phase)
            else:
                a = rng.normal(0, 1, d)

            states.append(s)
            actions.append(a)

            s_next, _, _, _ = env.step(a, noise_std=step_noise_std)
            next_states.append(s_next)

            s = s_next

    states = np.array(states).reshape(num_trajs, traj_length, d)
    actions = np.array(actions).reshape(num_trajs, traj_length, d)
    next_states = np.array(next_states).reshape(num_trajs, traj_length, d)

    train_data = {
        'states': states[:n_train],
        'actions': actions[:n_train],
        'next_states': next_states[:n_train]
    }
    val_data = {
        'states': states[n_train:n_train+n_val],
        'actions': actions[n_train:n_train+n_val],
        'next_states': next_states[n_train:n_train+n_val]
    }
    test_data = {
        'states': states[n_train+n_val:],
        'actions': actions[n_train+n_val:],
        'next_states': next_states[n_train+n_val:]
    }

    return train_data, val_data, test_data, env.A, env.B
