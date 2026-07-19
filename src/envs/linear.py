import numpy as np
from typing import Tuple

class LinearSystemEnv:
    def __init__(self, d: int = 8, target_L: float = 1.05, noise_std: float = 0.01, seed: int = 42):
        self.d = d
        self.target_L = target_L
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        A_init = self.rng.normal(0, 1, (d, d))

        U, S, Vt = np.linalg.svd(A_init)

        S = np.linspace(0.5 * target_L, target_L, d)
        self.A = U @ np.diag(S) @ Vt

        self.B = self.rng.normal(0, 0.5, (d, d))

        self.state = np.zeros(d)

    def reset(self) -> np.ndarray:
        self.state = self.rng.normal(0, 1, self.d)
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        noise = self.rng.normal(0, self.noise_std, self.d)
        self.state = self.A @ self.state + self.B @ action + noise

        return self.state.copy(), 0.0, False, {}

def generate_linear_dataset(
    target_L: float = 1.05,
    d: int = 8,
    num_transitions: int = 50000,
    traj_length: int = 100,
    seed: int = 42
):
    """
    Generates a dataset of trajectories for the linear system.
    Policy: 50% random Gaussian, 50% oscillating (sine waves).
    Returns (train_data, val_data, test_data) where each is a dict of 'states', 'actions', 'next_states'.
    """
    env = LinearSystemEnv(d=d, target_L=target_L, seed=seed)
    num_trajs = num_transitions // traj_length

    states, actions, next_states = [], [], []

    rng = np.random.default_rng(seed)

    for i in range(num_trajs):
        s = env.reset()
        is_oscillating = (i % 2 == 0)

        freq = rng.uniform(0.1, 1.0, d)
        phase = rng.uniform(0, 2*np.pi, d)

        for t in range(traj_length):
            if is_oscillating:
                a = np.sin(freq * t + phase)
            else:
                a = rng.normal(0, 1, d)

            states.append(s)
            actions.append(a)

            s_next, _, _, _ = env.step(a)
            next_states.append(s_next)

            s = s_next

    states = np.array(states).reshape(num_trajs, traj_length, d)
    actions = np.array(actions).reshape(num_trajs, traj_length, d)
    next_states = np.array(next_states).reshape(num_trajs, traj_length, d)

    n_train = int(0.8 * num_trajs)
    n_val = int(0.1 * num_trajs)

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

    return train_data, val_data, test_data, env.A
