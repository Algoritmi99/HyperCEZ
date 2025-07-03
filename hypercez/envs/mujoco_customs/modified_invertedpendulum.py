import os
import gymnasium
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from .perturbed_bodypart_env import ModifiedSizeEnvFactory

InvertedPendulumModifiedLengthEnv = lambda *args, **kwargs : ModifiedSizeEnvFactory(ModifiedInvertedPendulumEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/inverted_pendulum.xml", *args, **kwargs
)


class ModifiedInvertedPendulumEnv(InvertedPendulumEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 2, None)
        utils.EzPickle.__init__(self)

class InvertedPendulumBin(InvertedPendulumEnv):
    def __init__(self, offset=0):
        self.offset = offset
        InvertedPendulumEnv.__init__(self)

    def _get_obs(self):
        obs = super()._get_obs()
        obs[0] += self.offset
        return obs
