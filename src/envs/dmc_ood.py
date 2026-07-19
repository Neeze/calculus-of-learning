import numpy as np


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

    def _get_obs(self, time_step):
        obs = []
        for v in time_step.observation.values():
            obs.append(np.array(v).flatten())
        return np.concatenate(obs)

def make_dmc_env(domain_name: str, task_name: str, mode: str = "train", seed: int = 42):
    """
    Creates DMC environment with OOD shift if mode is 'test_ood'.
    """
    if mode == "train":
        return DMCOODWrapper(domain_name, task_name, seed=seed)
    elif mode == "test_ood":
        return DMCOODWrapper(domain_name, task_name, mass_shift=0.3, friction_shift=0.3, seed=seed)
    else:
        raise ValueError(f"Unknown mode: {mode}")
