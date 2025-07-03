import os

import gymnasium
import numpy as np
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env

from .gravity_envs import GravityEnvFactory
from .perturbed_bodypart_env import ModifiedSizeEnvFactory
from .wall_envs import SimpleWallEnvFactory, StairsFactory

HopperSimpleWallEnv = lambda *args, **kwargs : SimpleWallEnvFactory(ModifiedHopperEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/hopper.xml", ori_ind=-1, *args, **kwargs
)

HopperStairs = lambda *args, **kwargs : StairsFactory(ModifiedHopperEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/hopper.xml", ori_ind=-1, *args, **kwargs
)

HopperGravityEnv = lambda *args, **kwargs : GravityEnvFactory(ModifiedHopperEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/hopper.xml", *args, **kwargs
)

HopperModifiedBodyPartSizeEnv = lambda *args, **kwargs : ModifiedSizeEnvFactory(ModifiedHopperEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/hopper.xml", *args, **kwargs
)


class ModifiedHopperEnv(HopperEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4, None)
        utils.EzPickle.__init__(self)
        super().__init__(**kwargs)


class HopperWithSensorEnv(HopperEnv, utils.EzPickle):
    """
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we get sensor readouts with distances to the wall
    """

    def __init__(self, n_bins=10, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4, None)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        obs = np.concatenate([
            HopperEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs
