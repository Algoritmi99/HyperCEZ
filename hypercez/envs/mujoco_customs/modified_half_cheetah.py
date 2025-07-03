import os

import gymnasium
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from hypercez.envs.half_cheetah import HalfCheetahEnv

from .gravity_envs import GravityEnvFactory
from .perturbed_bodypart_env import ModifiedSizeEnvFactory
from .wall_envs import WallEnvFactory

HalfCheetahWallEnv = lambda *args, **kwargs : WallEnvFactory(ModifiedHalfCheetahEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/half_cheetah.xml", ori_ind=-1, *args, **kwargs
)

HalfCheetahGravityEnv = lambda *args, **kwargs : GravityEnvFactory(ModifiedHalfCheetahEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/half_cheetah.xml", *args, **kwargs
)

HalfCheetahModifiedBodyPartSizeEnv = lambda *args, **kwargs : ModifiedSizeEnvFactory(ModifiedHalfCheetahEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/half_cheetah.xml", *args, **kwargs
)

class ModifiedHalfCheetahEnv(HalfCheetahEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 5, None)
        utils.EzPickle.__init__(self)

class HalfCheetahWithSensorEnv(HalfCheetahEnv, utils.EzPickle):
    """
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we get sensor readouts with distances to the wall
    """

    def __init__(self, n_bins=10, **kwargs):
        self.n_bins = n_bins
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 5, None)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        obs = np.concatenate([
            HalfCheetahEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs
