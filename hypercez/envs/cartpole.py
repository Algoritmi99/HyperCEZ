import os
import tempfile

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv

import xml.etree.ElementTree as ET


class CartpoleEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, model_path=None, pendulum_length=0.6, **kwargs):
        if model_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            model_path = f"{dir_path}/assets/cartpole.xml"

        self.pendulum_length = pendulum_length

        utils.EzPickle.__init__(self, model_path, pendulum_length)
        MujocoEnv.__init__(self, model_path, 2, None, render_mode=kwargs.get('render_mode'))

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = self.pendulum_length
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, self.pendulum_length]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        terminated = False
        truncated = False
        info = {}

        return ob, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(0, 0.1, size=self.init_qpos.shape)
        qvel = self.init_qvel + self.np_random.normal(0, 0.1, size=self.init_qvel.shape)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ob = self.reset_model()
        info = {}
        return ob, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def _get_ee_pos(self, x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - self.pendulum_length * np.sin(theta),
            -self.pendulum_length * np.cos(theta)
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


class CartpoleBinEnv(CartpoleEnv):
    def __init__(self, offset=0):
        self.offset = offset
        super().__init__()

    def _get_obs(self):
        obs = super()._get_obs()
        obs[0] += self.offset
        return obs

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = self.pendulum_length
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([self.offset, self.pendulum_length]))) / (
                        cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        terminated = False
        truncated = False
        info = {}

        return ob, reward, terminated, truncated, info


class CartpoleLengthEnv(CartpoleEnv, utils.EzPickle):
    def __init__(
            self,
            body_parts=["cpole"],
            length=0.6,
            *args,
            **kwargs
    ):
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'cartpole.xml')

        # Modify specified body parts in the XML
        tree = ET.parse(model_path)
        for body_part in body_parts:
            geom = tree.find(f".//geom[@name='{body_part}']")
            if geom is None:
                raise ValueError(f"Body part '{body_part}' not found in the model.")
            fromto = [float(x) for x in geom.attrib["fromto"].split()]
            fromto[-1] = -length  # Adjust the length
            geom.attrib["fromto"] = " ".join(map(str, fromto))

        # Write modified XML to a temp file
        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        # Init base env with modified model and length
        super().__init__(model_path=file_path, pendulum_length=length, **kwargs)

        # Proper EzPickle call for Gymnasium
        utils.EzPickle.__init__(self, body_parts, length, *args, **kwargs)


if __name__ == "__main__":
    env = CartpoleLengthEnv(length=1.2, render_mode='human')
    env.reset()
    while True:
        env.render()
        x = env.step(env.action_space.sample())
