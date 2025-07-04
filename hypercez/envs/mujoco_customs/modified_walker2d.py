import numpy as np
import os
import gymnasium

from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env

from .wall_envs import WallEnvFactory
from .gravity_envs import GravityEnvFactory
from .perturbed_bodypart_env import ModifiedSizeEnvFactory

from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv

Walker2dWallEnv = lambda *args, **kwargs : WallEnvFactory(ModifiedWalker2dEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/walker2d.xml", ori_ind=-1, *args, **kwargs
)

Walker2dGravityEnv = lambda *args, **kwargs : GravityEnvFactory(ModifiedWalker2dEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/walker2d.xml", *args, **kwargs
)

Walker2dModifiedBodyPartSizeEnv = lambda *args, **kwargs : ModifiedSizeEnvFactory(ModifiedWalker2dEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/walker2d.xml", *args, **kwargs
)


class ModifiedWalker2dEnv(Walker2dEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4)
        utils.EzPickle.__init__(self)

class Walker2dWithSensorEnv(Walker2dEnv, utils.EzPickle):
    """
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we get sensor readouts with distances to the wall
    """

    def __init__(self, n_bins=10, **kwargs):
        self.n_bins = n_bins
        super().__init__(**kwargs)
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4, None)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        obs = np.concatenate([
            Walker2dEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs
