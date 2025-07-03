import os

import gymnasium
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.envs.mujoco.ant_v4 import AntEnv

from .gravity_envs import GravityEnvFactory
from .wall_envs import MazeFactory

AntGravityEnv = lambda *args, **kwargs : GravityEnvFactory(ModifiedAntEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/ant.xml", *args, **kwargs
)


AntMaze = lambda *args, **kwargs : MazeFactory(ModifiedAntEnv)(
    model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/ant.xml", ori_ind=0, *args, **kwargs
)




class ModifiedAntEnv(AntEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4, None)
        utils.EzPickle.__init__(self)
