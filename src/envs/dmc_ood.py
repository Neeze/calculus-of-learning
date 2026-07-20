import numpy as np


# E2 OOD grid (spec §3.1): single-axis shifts in both signs plus two combined
# corners. Keys are condition names used in cached dataset filenames and
# results.json.
OOD_GRID = {
    "mass_m30": {"mass_shift": -0.30, "friction_shift": 0.0},
    "mass_m15": {"mass_shift": -0.15, "friction_shift": 0.0},
    "mass_p15": {"mass_shift": 0.15, "friction_shift": 0.0},
    "mass_p30": {"mass_shift": 0.30, "friction_shift": 0.0},
    "fric_m30": {"mass_shift": 0.0, "friction_shift": -0.30},
    "fric_m15": {"mass_shift": 0.0, "friction_shift": -0.15},
    "fric_p15": {"mass_shift": 0.0, "friction_shift": 0.15},
    "fric_p30": {"mass_shift": 0.0, "friction_shift": 0.30},
    "comb_p30": {"mass_shift": 0.30, "friction_shift": 0.30},
    "comb_m30": {"mass_shift": -0.30, "friction_shift": -0.30},
}


class DMCOODWrapper:
    """
    Wraps a dm_control environment and applies physics shifts.
    """
    def __init__(self, domain_name: str, task_name: str, mass_shift: float = 0.0, friction_shift: float = 0.0, seed: int = 42):
        from dm_control import suite
        self.env = suite.load(domain_name, task_name, task_kwargs={'random': seed})
        self.mass_shift = mass_shift
        self.friction_shift = friction_shift
        self._apply_physics_shift()

    def _apply_physics_shift(self):
        """
        Modifies the body mass and geometry friction in the MuJoCo model.
        """
        if self.mass_shift != 0.0:
            self.env.physics.model.body_mass[:] *= (1.0 + self.mass_shift)

        if self.friction_shift != 0.0:
            self.env.physics.model.geom_friction[:] *= (1.0 + self.friction_shift)

    def reset(self):
        time_step = self.env.reset()
        return self._get_obs(time_step)

    def step(self, action):
        time_step = self.env.step(action)
        return self._get_obs(time_step), time_step.reward, time_step.last(), {}

    def get_physics_state(self):
        """Ground-truth (qpos, qvel) for the linear probe (spec §4b.3)."""
        physics = self.env.physics
        return np.concatenate([
            np.array(physics.data.qpos).flatten(),
            np.array(physics.data.qvel).flatten(),
        ])

    def set_physics_state(self, qpos, qvel):
        """Set simulator state directly. Used by the shift-strength oracle (spec §6.2)."""
        physics = self.env.physics
        with physics.reset_context():
            physics.data.qpos[:] = qpos
            physics.data.qvel[:] = qvel

    def action_spec(self):
        return self.env.action_spec()

    def _get_obs(self, time_step):
        obs = []
        for v in time_step.observation.values():
            obs.append(np.array(v).flatten())
        return np.concatenate(obs)


def make_dmc_env(domain_name: str, task_name: str, mode: str = "train", seed: int = 42,
                 mass_shift: float = None, friction_shift: float = None):
    """
    Creates DMC environment. Modes:
      - "train": default physics.
      - "test_ood": legacy single-point shift (+30% mass, +30% friction).
      - a key of OOD_GRID (e.g. "mass_p30"): that grid condition.
    Explicit mass_shift/friction_shift override the mode's shifts.
    """
    if mode == "train":
        shifts = {"mass_shift": 0.0, "friction_shift": 0.0}
    elif mode == "test_ood":
        shifts = {"mass_shift": 0.3, "friction_shift": 0.3}
    elif mode in OOD_GRID:
        shifts = dict(OOD_GRID[mode])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mass_shift is not None:
        shifts["mass_shift"] = mass_shift
    if friction_shift is not None:
        shifts["friction_shift"] = friction_shift

    return DMCOODWrapper(domain_name, task_name, seed=seed, **shifts)
