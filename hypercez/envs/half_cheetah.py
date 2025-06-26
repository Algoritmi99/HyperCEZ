import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        self.xposbefore = None

        utils.EzPickle.__init__(self, render_mode)

        super().__init__(
            model_path='half_cheetah.xml',
            frame_skip=5,
            observation_space=None,  # auto-computed from _get_obs()
            render_mode=render_mode,
        )

    def step(self, action):
        self.xposbefore = self.data.qpos[0]  # self.data is a shortcut to self.mujoco_sim.data
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - self.xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        terminated = False
        truncated = False
        info = {"reward_run": reward_run, "reward_ctrl": reward_ctrl}

        return ob, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([
            (self.data.qpos.flat[:1] - self.xposbefore) / self.dt,
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.random(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.xposbefore = np.copy(self.data.qpos[0])
        return self._get_obs()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ob = self.reset_model()
        info = {}
        return ob, info

    def viewer_setup(self):
        if self.viewer is not None:
            self.viewer.cam.distance = self.model.stat.extent * 0.5


if __name__ == "__main__":
    env = HalfCheetahEnv(render_mode='human')
    env.reset()
    while True:
        env.render()
        x = env.step(env.action_space.sample())